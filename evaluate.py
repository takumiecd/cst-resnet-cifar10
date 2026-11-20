"""Evaluate a released checkpoint on the official CIFAR-10 test split."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchcst.hub import from_pretrained, load_safetensors
from torchvision.datasets import CIFAR10

from infer import HUB
from model import Classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "cst-one", "cst-two"), default="cst-two")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local", type=Path)
    args = parser.parse_args()
    path = args.local / "config.json" if args.local else Path(
        hf_hub_download(HUB, "config.json", revision=args.revision, token=False)
    )
    config = json.loads(path.read_text())
    factory = partial(Classifier, **config["variants"][args.variant])
    filename = f"{args.variant}.safetensors"
    if args.local:
        inst = load_safetensors(args.local / filename, model_factory=factory, device=args.device)
    else:
        inst = from_pretrained(
            HUB, model_factory=factory, filename=filename, device=args.device,
            revision=args.revision, token=False,
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cifar = CIFAR10(args.data, train=False, download=args.download)
    images = torch.from_numpy(cifar.data)
    labels = torch.tensor(cifar.targets, device=args.device)
    mean = torch.tensor(config["channel_mean"], device=args.device)
    std = torch.tensor(config["channel_std"], device=args.device)
    correct = 0
    with torch.inference_mode():
        for batch, target in zip(images.split(256), labels.split(256), strict=True):
            x = (batch.to(args.device, dtype=torch.float32) * (1.0 / 255.0) - mean) / std
            correct += int((inst(x.permute(0, 3, 1, 2)).argmax(1) == target).sum())
    print(json.dumps({"correct": correct, "examples": len(labels), "accuracy": correct / len(labels)}))


if __name__ == "__main__":
    main()
