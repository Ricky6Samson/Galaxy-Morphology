
# Galaxy Morphology Classification — 6-Class Experiment

## Overview

This experiment investigates whether combining visually similar galaxy morphology categories can improve classification performance using deep learning.

It builds upon the original 10-class Galaxy10 DECaLS classifier and explores a broader six-class classification scheme. The experiment was motivated by feedback regarding confusion between visually similar morphological subcategories.

The objective was to evaluate whether broader, more distinguishable morphological categories could improve model performance while retaining meaningful astronomical classifications.

## Dataset

**Galaxy10 DECaLS**

- Total images: 17,736
- Original categories: 10
- Revised categories: 6
- Training/validation split: 80:20
- Stratified sampling with random state 42

### Class Restructuring

| New Class | Original Categories Combined | Images |
|---|---|---:|
| Disturbed | Disturbed | 1,081 |
| Merging | Merging | 1,853 |
| Elliptical | Round Smooth, In-between Round Smooth, Cigar Shaped Smooth | 5,006 |
| Barred Spiral | Barred Spiral | 2,043 |
| Unbarred Spiral | Tight Spiral, Loose Spiral | 4,457 |
| Edge-on | Edge-on with Bulge, Edge-on without Bulge | 3,296 |

## Model Architecture

**Transfer Learning — EfficientNetB0**

- Pretrained EfficientNetB0 backbone (ImageNet weights)
- Global Average Pooling
- Dense layer (256 units, ReLU)
- Dropout (0.4)
- Softmax output layer (6 classes)

The original model architecture was retained, with only the final classification layer modified to accommodate six output classes.

## Training Strategy

Training was conducted in three stages:

### Phase 1 — Feature Extraction
- EfficientNetB0 backbone frozen.
- Classification head trained using pretrained features.

### Phase 2 — Fine-Tuning
- Top 40 layers of the backbone unfrozen.
- Fine-tuning performed using a lower learning rate.
- Training continued across two runs.

### Phase 3 — Data Augmentation
- Data augmentation introduced during fine-tuning.
- Training continued with a reduced learning rate to improve generalization.

The experiment retained the original training methodology as much as possible to make the comparison meaningful.

## Results

| Training Stage | Validation Accuracy | Validation Loss |
|---|---:|---:|
| Phase 1 | 72.89% | 0.7588 |
| Phase 2 — Run 1 | 77.54% | 0.6544 |
| Phase 2 — Run 2 | 78.49% | 0.6152 |
| Phase 3 — Augmented | **78.69%** | **0.6096** |

The final model achieved a validation accuracy of **78.69%**, with a validation loss of **0.6096**.

## Final Model

**Model:** `galaxy10_efficientnetb0_phase3.keras`

The Phase 3 checkpoint was selected as the final model based on its validation performance.

## Evaluation

Further evaluation includes:

- Confusion matrix
- Per-class precision and recall
- Macro F1-score
- Grad-CAM visualizations to investigate the image regions influencing model predictions

## Key Takeaway

This experiment explores how astronomical class definitions influence the performance of automated galaxy morphology classification.

By consolidating visually similar categories, the experiment achieved improved validation performance compared with the original 10-class model. It also provides a foundation for further investigation into more physically informative classification features, including spectral information.
