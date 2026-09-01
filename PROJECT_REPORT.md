Project Documentation: Layered Representations from a Single Image

**Course:** Deep Learning for Computer Vision (DLCV)  
**Student / GitHub user:** 22f2000506  
**Date:** 1 September 2026

## Abstract

This project converts a single RGB image into a re-composable stack of layers. A semantic segmentation model identifies interpretable regions, while a monocular depth model estimates their relative ordering. The system exports each region as an RGBA PNG, lists layers from near to far in a JSON manifest, and reconstructs the scene by alpha compositing the layers. It also exports luminance-based albedo and shading proxies as an exploratory intrinsic-appearance split.

## 1. Problem and motivation

Standard bitmap images flatten all scene content into one RGB array. This makes object-wise editing, simple parallax animation, and scene-aware processing difficult. The goal is to infer a useful layered representation from only one image, despite occlusion and the absence of true 3D information.

## 2. Proposed approach

The pipeline has four stages.

1. **Semantic grouping.** Mask2Former predicts semantic regions such as person, vehicle, furniture, wall, floor, sky, and vegetation.
2. **Depth estimation.** Depth Anything V2 predicts dense relative depth. It is not interpreted as metric distance.
3. **Layer construction.** For each accepted semantic mask, the original RGB image is copied into an RGBA image with the mask as alpha. The average depth inside the mask determines near-to-far ordering.
4. **Appearance proxy.** RGB luminance is retained as a shading proxy. RGB divided by luminance is used as a chromatic albedo proxy. This is a stretch feature and is not claimed to be ground-truth intrinsic decomposition.

When masks overlap, pixels are assigned only once, beginning with the highest-confidence mask. This makes recomposition deterministic.

## 3. Implementation

The implementation is in `src/layerify/pipeline.py` and is invoked through `run.py`. It uses Python, Pillow, NumPy, PyTorch, and Hugging Face Transformers. The primary configuration uses `facebook/mask2former-swin-small-ade-semantic` for segmentation and `depth-anything/Depth-Anything-V2-Small-hf` for depth.

Each run writes a directory containing the layers, `relative_depth.png`, `composite.png`, and `manifest.json`. The manifest records labels, confidences, mean relative depth, pixel counts, and filenames. An offline heuristic backend supports smoke testing but is not used as the main result because it cannot provide semantic recognition.

## 4. Experimental protocol

Evaluate on 10-20 images covering at least indoor and outdoor scenes. Include scenes with multiple object categories and depth variation. Use the same minimum-area threshold initially, then report any manual adjustment.

For each example, capture:

- the original image;
- the relative-depth image;
- the near-to-far RGBA layer stack;
- the reconstructed composite; and
- one albedo/shading proxy pair.

## 5. Results



## 6. Discussion and limitations

The method is useful when the image contains recognizable semantic categories and clear depth cues. Its strengths are interpretable output files and direct recomposability. Its main limitations are imperfect segmentation boundaries, depth-order ambiguity for overlapping or reflective surfaces, and model bias inherited from pretrained data. Monocular depth is relative rather than metric. The appearance decomposition is intentionally described as a proxy: a real intrinsic-image method would require stronger illumination modeling and evaluation data.

## 7. Conclusion

The project demonstrates that pretrained semantic segmentation and monocular depth estimation can be combined into a practical single-image layered representation. The exported RGBA files, ordering manifest, depth map, and recomposition provide concrete evidence for the brief's requirements. Future work would add instance-aware splitting, inpainting behind removed foreground objects, and learned intrinsic decomposition.

## References

1. Cheng, B. et al. *Masked-attention Mask Transformer for Universal Image Segmentation* (CVPR 2022).
2. Yang, L. et al. *Depth Anything V2* (2024).
3. Hugging Face Transformers documentation: image segmentation and depth estimation pipelines.
