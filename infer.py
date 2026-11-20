"""Classify one image using the CIFAR-10 model and published weights."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchcst.hub import from_pretrained, load_safetensors

from model import Classifier


HUB = "takumiecd/cst-resnet-cifar10"


def image_tensor(path, config, device):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.to(device=device, dtype=torch.float32)
    pixels = pixels.view(1, 32, 32, 3) * (1.0 / 255.0)
    mean = pixels.new_tensor(config["channel_mean"])
    std = pixels.new_tensor(config["channel_std"])
    return ((pixels - mean) / std).permute(0, 3, 1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--variant", choices=("baseline", "cst-one", "cst-two"), default="cst-two")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local", type=Path, help="Directory with config.json and Safetensors")
    args = parser.parse_args()
    config_path = args.local / "config.json" if args.local else Path(
        hf_hub_download(HUB, "config.json", revision=args.revision, token=False)
    )
    config = json.loads(config_path.read_text())
    filename = f"{args.variant}.safetensors"
    factory = partial(Classifier, **config["variants"][args.variant])
    if args.local:
        model = load_safetensors(args.local / filename, model_factory=factory, device=args.device)
    else:
        model = from_pretrained(
            HUB, filename=filename, model_factory=factory, device=args.device,
            revision=args.revision, token=False,
        )
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        values, indices = model(image_tensor(args.image, config, args.device)).softmax(1)[0].topk(3)
    print(json.dumps([
        {"label": config["labels"][int(index)], "probability": float(value)}
        for value, index in zip(values, indices, strict=True)
    ], indent=2))


if __name__ == "__main__":
    main()
