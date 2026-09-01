"""Command-line runner for the DLCV project."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from layerify import LayeredRepresentationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic, depth-ordered RGBA layers from one image.")
    parser.add_argument("input", help="Path to an RGB image")
    parser.add_argument("--output", default="output", help="Folder for layer images and manifest")
    parser.add_argument("--backend", choices=["transformers", "heuristic"], default="transformers")
    parser.add_argument("--min-area", type=float, default=0.01, help="Minimum fraction of image pixels per layer")
    args = parser.parse_args()
    manifest = LayeredRepresentationPipeline(args.backend, args.min_area).run(args.input, args.output)
    print(f"Created {len(manifest['layers'])} layers in {args.output}")


if __name__ == "__main__":
    main()
