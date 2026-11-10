# November 2026 CID experiment

The [release assets](https://github.com/shumao469/RBCM/releases/tag/cid-2026-11-10)
provide the original six RBCM HED-lite checkpoints and the frozen HED
configuration for each experimental arm. The context correction itself is
deterministic. The HED experiment uses the same author-supplied BSDS checkpoint
in every arm, with no fine-tuning.

## HED inference with the evaluated configurations

From the repository root on Linux or WSL, using Python 3.10–3.12:

```bash
python -m pip install "numpy==1.26.4" "opencv-python-headless==4.12.0.88" "Pillow==11.3.0" --no-deps
curl -fL https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/deploy.prototxt -o deploy.prototxt
curl -fL https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel -o hed_pretrained_bsds.caffemodel
python experiments/cid_20261110/infer_hed.py image.png --deploy deploy.prototxt --weights hed_pretrained_bsds.caffemodel --output maps.npz
```

The command verifies the graph and checkpoint hashes and exports one native-size
probability map per arm before nonmaximum suppression or thresholding.
Use `--protocol nonzero` for the sensitivity analysis. The corresponding locked
decision thresholds are recorded in [hed_source_locked.json](hed_source_locked.json).
Exact HED provenance, including the source URLs and expected SHA-256 digests, is
recorded in [hed_provenance.json](hed_provenance.json).

The original [TEED BIPED checkpoint](https://raw.githubusercontent.com/xavysp/TEED/40fa4b1391dc6424f88989d0ca75d5b592c8681d/checkpoints/BIPED/5/5_model.pth)
has SHA-256 `0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba`.
The EdgeAnchor NYUDv2 run uses `pretrained/nyudv2_strict/best.pt` from the RBCM
checkpoint archive.

## Historical benchmark runners

The scripts below preserve the exact bytes used in the completed jobs. They
retain the original WSL directory layout, receipt hashes and cross-job cache
dependencies. Full benchmark replay requires the original data, admission
manifests, source splits, cached receipts, and the pinned one-to-one BSDS
matcher, including the local seeded extension. The inference command above
works with independently downloaded HED assets and a user-supplied image.

- [CID AP/ODS/OIS and standard HED](../../work/cid_ap_hed_20261109/)
- [Source selection, HED+RBCM evaluation and timing](../../work/hed_rbcm_20261110/)
- [Nonzero source selection and target evaluation](../../work/hed_nonzero_20261110/)
- [Shared transfer and benchmark dependencies](../../work/validation_20261108/engineering/)
- [Context operators and frozen backbone runtime](../../work/round7_20261031/)
