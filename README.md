# Harris Corner Detector from Scratch

A from-scratch NumPy implementation of the Harris corner detector with separable Gaussian filtering, eigenvalue analysis, and invariance experiments — benchmarked against OpenCV.

---

## Overview

This project implements core computer vision filtering primitives and the Harris corner detector **without relying on OpenCV's corner detection routines**. It demonstrates:

- 2D convolution with configurable padding (`reflect` / `zero`)
- 1D separable convolution and Gaussian separability (with speedup verification)
- Harris corner response computation using Gaussian or box windowing
- Vectorized 3×3 non-maximum suppression
- Eigenvalue-based classification of flat / edge / corner regions
- Repeatability experiments under rotation, scale, and brightness changes
- Parameter sweep (`sigma`, `k`, window type) and runtime comparison vs. `cv2.cornerHarris`

---
<img width="1788" height="880" alt="image" src="https://github.com/user-attachments/assets/efb92a64-2e25-4aaf-8c0a-1a7b31f3de22" />


## Repository Structure
.
├── harris_corner_detector.py # Main script
├── CHECKERBOARD.png # Test image (required)
├── LIBRARY.png # Test image (required)
├── NATURAL.png # Test image (required)
├── FOGGY.png # Test image (required)
└── README.md



## Requirements
numpy
opencv-python
matplotlib



Install with:

```bash
pip install numpy opencv-python matplotlib
Input Images
The script expects four grayscale test images. Place them in the same directory as the script, or upload them to /content/ (Colab) or /content/drive/MyDrive/.

Filename	Purpose
CHECKERBOARD.png	Ideal corner structure, used for separability & correctness checks
LIBRARY.png	Structured scene with strong edges/corners
NATURAL.png	Textured natural image
FOGGY.png	Low-contrast image to test threshold sensitivity
Images are automatically resized to 256×256 float32.

If a file is missing, the script raises a FileNotFoundError with the searched paths.

Usage
Local
bash
python harris_corner_detector.py
Google Colab
Upload the four images to /content/ (or mount Drive).

Run the script.

The script runs all tasks sequentially and produces printed metrics plus Matplotlib figures.

What Each Task Does
Task 1 — Filtering Foundations
convolve2d(image, kernel, padding) — naive 2D correlation (kernel is flipped internally to implement true convolution).

convolve1d(image, kernel1d, axis, padding) — 1D separable convolution along rows or columns.

separable_gaussian_filter(...) — applies row then column 1D kernels.

get_gaussian_kernel(sigma) — normalized Gaussian with radius ceil(3σ).

Verification: for σ ∈ {1.0, 2.0, 4.0}, compares full 2D Gaussian convolution vs. separable filtering. Prints the speedup factor and max pixel difference (expected ≈ 0, confirming separability).

Task 2 — Harris Detector
Computes Ix, Iy via Sobel operators (normalized by 1/8).

Builds the structure tensor with Gaussian (default) or box window.

Response: R = det(M) − k·trace(M)².

Vectorized 3×3 NMS using a padded array and 8 neighbor comparisons.

Threshold applied as a ratio of max(R).

Outputs a 2×4 figure: corner overlays (top row) and R-response heatmaps (bottom row) for all four images.

Task 3 — Eigenvalue Analysis
Recovers λ₁ ≥ λ₂ analytically from Sxx, Syy, Sxy.

Prints identity checks (det = λ₁λ₂, trace = λ₁+λ₂) as numerical error.

Plots points in eigenvalue space classified as flat / edge / corner, overlaid with the theoretical Harris R = 0 boundary lines for k = 0.04 and k = 0.06:

text
λ₂ = r · λ₁,   r = (1 − 2k − √(1 − 4k)) / (2k)
Task 4 — Invariance Experiments
Measures repeatability = fraction of corners in image A that map (via known transform) within 2 pixels of a detected corner in image B.

Transform	Sweep
Rotation	0°–180° in 15° steps
Scale	0.5×–2.0× (8 values)
Brightness	+0 to +80 in steps of 10
Results plotted as three side-by-side line charts. Expect:

Rotation: high at 0°/180°, degrades gradually (Harris is not rotation-invariant in the strict sense due to discretization, but fairly stable).

Scale: drops off quickly — Harris is not scale-invariant.

Brightness: additive shift should leave gradients unchanged, so repeatability should remain high.

Task 5 — Parameter Study & OpenCV Comparison
Table of corner counts for sigma ∈ {1.0, 1.5, 2.0} × k ∈ {0.02, 0.04, 0.06} × {Gaussian, Box} window on LIBRARY.png.

Times the custom detector vs. cv2.cornerHarris on CHECKERBOARD.png and reports corner counts.

Implementation Notes
Convolution vs. correlation: convolve2d and convolve1d flip the kernel so the operation is a true convolution (Sobel response is unaffected in magnitude by this choice).

Padding: reflect is the default; zero is supported for 2D convolution.

Kernel shape: odd-sized only (asserted).

NMS: threshold is compared to max(R); if max(R) ≤ 0, threshold is set to 0.

Fixed threshold hack: get_corners(img, use_fixed) converts an absolute threshold into the equivalent ratio using a first pass at threshold_ratio=0.0.

The eigenvalue classification thresholds (l1, l2 ≥ 2.0, |λ₁ − λ₂| ≤ 15.0) are tuned for this dataset's response scale; adjust if using different image sizes or contrast.

Known Limitations
Not scale-invariant — as expected for plain Harris (see Task 4).

Runtime: pure-Python loops over kernel elements make 2D convolution slow for large kernels; the separable path is much faster. The runtime comparison against OpenCV is illustrative, not competitive.

Fixed thresholds in Task 3 classification are heuristic.

NMS uses strict >= in all 8 directions, which can suppress plateaus of equal response — acceptable for corner detection.

Images are hard-resized to 256×256, which changes aspect ratio; this is fine for the exercises but should be adapted for real pipelines.

Suggested Extensions
Add sub-pixel corner refinement (quadratic fit on R).

Implement multi-scale Harris (scale-space NMS) to address Task 4's scale sensitivity.

Replace loops with np.lib.stride_tricks.sliding_window_view or FFT-based convolution.

Add Shi–Tomasi (min(λ₁, λ₂)) response for comparison.

Evaluate repeatability against ground-truth homographies on a real dataset (e.g., HPatches).

License
This project is released under the MIT License. See LICENSE for details.

Test images should be replaced with public-domain or self-captured equivalents before public redistribution.

