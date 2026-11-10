# RBCM Edge Detection

The November 2026 extension evaluates source-selected contextual corrections on
frozen detectors. It includes the CID comparison, original HED and HED+RBCM
under the same backbone, and a separate nonzero-amplitude sensitivity analysis.

- [Latest experiment code, model provenance and inference](experiments/cid_20261110/)
- [Pretrained checkpoints and source-locked model configurations](https://github.com/shumao469/RBCM/releases/tag/cid-2026-11-10)
- [Original HED-lite / MEA reproduction release](https://github.com/shumao469/RBCM/tree/release), maintained by the [upstream RBCM-Edge project](https://github.com/liukaiming6563/RBCM-Edge/tree/release)

The context layer uses a calibrated detector score and normalized input Sobel
evidence. The source-domain protocol locks correction strength and deployment
threshold before evaluation on the overlap-screened CID test. Zero amplitude
returns the calibrated score. The nonzero analysis selects only from 0.5 and 1
and inherits the original HED, calibrated, and legacy controls.
