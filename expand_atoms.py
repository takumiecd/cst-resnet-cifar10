"""Rebuild the portable classifier weights from the compact physical Atoms."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torchcst import CSTLinear, ChebyshevSpec, chart_presets, kernels


LAYERS = {"stem": (2048, 128), "recurrent": (2048, 2048)}


def expand(atoms, outputs, inputs):
    kernel = kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8), base_tile_sites=1, merge_factor=2),
        coordinates=kernels.physical(),
    )
    charts = tuple(
        chart_presets.regular_grid(size, geometry="flat_torus") for size in (outputs, inputs)
    )
    linear = CSTLinear(charts=charts, atoms=atoms, kernel=kernel, backend="materialized")
    with torch.no_grad():
        return linear.operator.weight().contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atoms", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists at {args.output}")
    torch.set_num_threads(args.threads)
    state = load_file(args.atoms)
    for name, dimensions in LAYERS.items():
        key = f"{name}.physical_atoms"
        if key in state:
            state[f"{name}.weight"] = expand(state.pop(key), *dimensions)
    save_file(state, args.output, metadata={"format": "pt"})


if __name__ == "__main__":
    main()
