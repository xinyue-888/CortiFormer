# CortiFormer reproduction logic

This repository contains a **logic-level PyTorch draft** based on the manuscript *“CortiFormer: a cortical-continuity-aware real-time end-to-end detector for bone erosion in rheumatoid arthritis hand radiographs.”* It is intended to make the method easier to inspect and implement.

> **Status:** this is not the authors’ official source code, has not been run, and is not a drop-in training package. The RT-DETR integration, data loader, loss/metric adapters, and some formula-level details still need to be completed and checked against the final paper and dataset release.

## What is included

- `data_prep.py` — converts binary erosion masks to one-class detection boxes using 8-connected components; removes components smaller than 16 pixels.
- `cortiformer_logic.py` — draft modules for the WCP-Net backbone, LLWD, FDCP/SOCR/CFMM, DCUF, and HORep, plus an RT-DETR integration placeholder.
- `train_logic.py` — training configuration and high-level training, validation, test, and HORep folding flow.
- `requirements.txt` — framework versions reported in the manuscript and basic image-processing dependencies.

## Method at a glance

CortiFormer adapts RT-DETR-R18 for small bone erosion detection in hand radiographs:

1. **WCP-Net** replaces the backbone. Learnable lifting-wavelet downsampling is intended to retain fine cortical detail; FDCP blocks combine low-frequency context, high-frequency features, orientation continuity residuals (SOCR), and cross-frequency modulation (CFMM). The backbone exposes S3/S4/S5 features.
2. **DCUF** replaces the four cross-scale resampling-and-concatenation nodes in the RT-DETR hybrid encoder. It estimates a fused feature through two proximal-gradient unfolding stages, with semantic/detail consistency terms and a sparse proximal mapping.
3. **HORep** replaces RepConv blocks in RepC3. Training-time background statistics, prewhitened templates, and Hotelling prototypes are intended to be folded into a regular 3×3 convolution for inference.

The manuscript leaves AIFI, query selection, decoder, matching strategy, and detection losses unchanged.

## Dataset preparation

RAM-H1200 is not included. Obtain it from its official source and follow its terms. The conversion script assumes each official split has this layout:

```text
RAM-H1200/
  train/images/   train/masks/
  val/images/     val/masks/
  test/images/    test/masks/
```

Image and mask files are paired by filename stem, and nonzero mask pixels are treated as erosion. Check the official archive’s mask encoding and folder names before using the script; adjust `find_mask` or the mask threshold if needed. The manuscript specifies the official train/validation/test split and removal of connected components below 16 pixels.

Example conversion command (after installing the dependencies):

```bash
python data_prep.py --source /path/to/RAM-H1200 --output /path/to/converted-data
```

The result contains YOLO-format text labels and a `dataset.yaml`. It does not train a detector.

## Training details reported in the manuscript

- PyTorch 2.2, Ultralytics 8.3.78; RT-DETR-R18 initialized without pretrained weights.
- Resize with aspect ratio preserved, then pad to 640×640.
- AdamW, weight decay `1e-4`, 5-epoch linear warm-up followed by cosine decay, batch size 8, 300 epochs, and model EMA.
- Horizontal flip, brightness/contrast changes, random scale and translation within ±20%, and Mosaic; no vertical flip.
- WCP-Net channels `64/128/256/512`, stage depths `2/2/4/2`; SOCR lengths `7` and `11`; DCUF `K=2`; HORep `G=8`, `λ=2`, `T=3`, EMA momentum `0.99`, covariance shrinkage `0.1`. Freeze HORep statistics for the final 10 epochs before folding.
- Validation is used for checkpoint selection. Keep the test set for final evaluation. The paper reports means over seeds `0, 1, 2`.

The manuscript text available for this draft does not state the initial learning rate, and some displayed formulas did not survive text extraction. The `1e-4` value in `train_logic.py` is explicitly a placeholder, **not a paper-reported value**. Confirm all such details from the typeset equations or the authors before comparing results.

## Current limitations

The training script deliberately leaves dataset loading, RT-DETR graph replacement, the official criterion, validation metrics, and benchmark code as integration points. Several module details are engineering approximations where the manuscript’s prose did not define tensor dimensions or exact operators. Therefore, this repository documents the implementation logic; it does not currently reproduce the reported metrics.

## Citation

Hu D, Xie C, Zhao S, He X, Yang J, Wang W, Liu X, He Q, Chi K, Xie Z, Li Z. *CortiFormer: a cortical-continuity-aware real-time end-to-end detector for bone erosion in rheumatoid arthritis hand radiographs.* Cite the final published version or preprint with its DOI/URL when available.

## License

No license is included yet. Public visibility on GitHub does not by itself grant permission to reuse or redistribute the code. Add the license chosen by the rights holder before describing this repository as open-source software.
