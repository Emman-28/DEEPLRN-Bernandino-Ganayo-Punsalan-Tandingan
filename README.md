# FungAI: A Controlled Comparison of From-Scratch and Transfer-Learned Convolutional Networks for Microscopic Fungi Classification

Code and results for our paper comparing a from-scratch CNN against pretrained ResNet50 and EfficientNet-B0 backbones (frozen and fine-tuned) on the DeFungi microscopic fungi classification dataset, under one fixed data split and training recipe.

**Authors:** Bernandino, Ganayo, Punsalan, Tandingan

## What's in this repo

- `FungAI_Final.ipynb` — the complete notebook. Running it top to bottom reproduces every table and figure in the paper.

## Quick start

1. Open `FungAI_Final.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.**

That's it — no manual dataset download, no path editing. The first cells install `kagglehub` and pull the DeFungi dataset automatically; `DATA_DIR` is set from the returned path, so nothing needs to be edited by hand.

Total runtime for a full pass (7 main configurations + 10 seed-repeat runs) is approximately **5 hours** on a Colab T4. See [Runtime breakdown](#runtime-breakdown) below if you want to run a subset instead of everything.

## What the notebook does, section by section

| Section | Contents |
|---|---|
| 0. Setup | Installs dependencies, downloads DeFungi via `kagglehub`, sets the global seed (`SEED = 42`) |
| 1. Data loading & split | Stratified 70/15/15 split (`random_state=42`) → 6,379 train / 1,367 val / 1,368 test, matching Table 1 in the paper |
| 2. Transforms | Geometric-only augmentation (rotation, flips, resized crop) — no colour augmentation, since staining is diagnostic |
| 3. Baseline CNN | The four-block from-scratch architecture (Figure 1 in the paper) |
| 4. Pretrained backbones | ResNet50 / EfficientNet-B0 builders, with frozen vs. fine-tuned configuration logic |
| 5. Training loop | Adam optimiser, early stopping (patience 5, max 30 epochs) |
| 6. Evaluation | Accuracy, macro precision/recall/F1, confusion matrices |
| 7. Run experiments | Trains all 5 main configurations (baseline, ResNet50 × 2, EfficientNet-B0 × 2) — this is Table 3 |
| 8. Ablations | Augmentation on/off and weighted/unweighted loss, both on the from-scratch baseline — this is Table 5 |
| Statistical significance | Repeats all 5 main configurations across seeds 42 and 123 — this is Table 6 |
| 9. Results table | Aggregates everything above into a single summary table |
| 10. Grad-CAM | Runs Grad-CAM on every trained configuration, **after** all training is finished (see note below) — this produces Figure 3 |
| Save outputs | Zips and downloads every confusion matrix, Grad-CAM figure, and the results CSV before the Colab runtime can disconnect |

## Reproducing specific results

- **Table 3** (main comparison): run through Section 7.
- **Table 4** (per-class F1): included in Section 6's `evaluate_model` output for each config trained in Section 7.
- **Table 5** (ablations): Section 8.
- **Table 6** (seed variance): the "Statistical significance" section. This alone retrains all 5 configurations twice (10 runs) and is the most time-consuming part of a full run.
- **Figure 3** (Grad-CAM): Section 10.

## Implementation notes that matter for reproduction

- **Seeding:** Python, NumPy, and PyTorch are all seeded to 42 before the data split, so the split indices are exact and reproducible. Residual nondeterminism from cuDNN kernel selection remains — this is expected and is exactly why the paper reports variance across seeds (Table 6) rather than trusting single-run point estimates. As a concrete example: the baseline CNN scored 0.6567 macro F1 in the main run (Table 3) and 0.6518 when independently re-run at the same seed 42 as part of the seed-repeat experiment (Table 6).
- **Grad-CAM ordering:** Grad-CAM is run in its own pass, after every training run is complete — not interleaved between training runs. Drawing a batch from the test loader advances the global random number stream, which would otherwise shift the next model's initialisation if Grad-CAM ran between training calls.
- **Grad-CAM on frozen backbones:** frozen configurations carry no gradient into the backbone by default (`requires_grad=False` on all pretrained parameters). `run_gradcam` explicitly sets `requires_grad_(True)` on the input tensor so activations stay differentiable even when the backbone itself is frozen.
- **Sampled batch:** the Grad-CAM figure draws one batch from the unshuffled test loader. Because `ImageFolder` iterates classes in sorted directory order, this batch is entirely class H1. Figure 3 therefore compares attention across configurations on a fixed class, not across classes.

## Runtime breakdown

Measured on a Colab T4 GPU, from actual per-epoch timing:

| Stage | Approx. time |
|---|---|
| 7 main configurations (5 main + 2 ablations) | ~107 minutes |
| 10 seed-repeat runs (Statistical significance section) | ~3.2 hours |
| **Total, full notebook** | **~5 hours** |

Per-epoch time ranges from about 35 to 63 seconds depending on configuration. If you only need the headline numbers (Table 3), you can stop after Section 7 (~2 hours including ablations) and skip the seed-repeat section.

## Known limitations (see paper Section 6.2 for full discussion)

- The train/val/test split is stratified at the image level, not grouped by source photograph. DeFungi images are crops from a smaller number of source photographs, so crops from the same photograph can appear in both train and test — this likely inflates the absolute numbers reported here. Treat Table 3 as an optimistic upper bound; relative ordering between configurations is more robust than absolute values.
- Only two seeds are used for variance estimation, which bounds Table 6 loosely — enough to check whether a gap could be noise, not enough for a precise confidence interval.
- Ablations (Section 8) vary augmentation and loss weighting on the from-scratch baseline only, not on the pretrained backbones.

## Requirements

Everything installs automatically in the first cells:
```
kagglehub
torch, torchvision
scikit-learn
pytorch-grad-cam
pandas, numpy, matplotlib, seaborn
```

No `requirements.txt` is needed for Colab; if running locally, install the above via pip and ensure a CUDA-capable GPU is available (the notebook falls back to CPU automatically but will be substantially slower).

## Citation

If you use this code, please cite:
```
Bernandino, Ganayo, Punsalan, Tandingan. "FungAI: A Controlled Comparison of
From-Scratch and Transfer-Learned Convolutional Networks for Microscopic
Fungi Classification." 2026.
```

## Dataset

DeFungi (Sopo et al., 2021), 9,114 labelled microscopy images across five fungal morphology classes, originally hosted on the UCI Machine Learning Repository. This notebook pulls it via the Kaggle mirror (`joebeachcapital/defungi`) through `kagglehub` — no manual download required.
