from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from layerify import LayeredRepresentationPipeline


def test_heuristic_backend_exports_reconstructable_layers(tmp_path):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:20, :30] = (30, 80, 200)
    image[:20, 30:] = (210, 190, 20)
    image[20:, :30] = (10, 160, 80)
    image[20:, 30:] = (180, 40, 120)
    source = tmp_path / "input.png"
    Image.fromarray(image).save(source)

    manifest = LayeredRepresentationPipeline("heuristic", min_area=0.001).run(source, tmp_path / "result")
    assert len(manifest["layers"]) >= 2
    assert (tmp_path / "result" / "manifest.json").exists()
    assert (tmp_path / "result" / "composite.png").exists()
    orders = [layer["order_near_to_far"] for layer in manifest["layers"]]
    assert orders == list(range(1, len(orders) + 1))
