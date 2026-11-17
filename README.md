
# Galaxy Morphology Classification — 6-Class Experiment

## 1. Project Overview

This experiment explores automated galaxy morphology classification using deep learning and the Galaxy10 DECaLS dataset.

Following feedback on the original 10-class classification approach, the experiment investigated whether grouping closely related morphological categories into broader classes could produce a more effective classification task.

The original ten categories were reorganised into six broader morphological groups. An EfficientNetB0-based convolutional neural network was subsequently trained and fine-tuned to classify these groups.

## 2. Dataset

- **Dataset:** Galaxy10 DECaLS
- **Total images:** 17,736
- **Original categories:** 10
- **Revised categories:** 6
- **Train-validation split:** 80:20
- **Splitting method:** Stratified sampling
- **Random state:** 42

### Class Restructuring

| Revised Class | Original Galaxy10 Categories | Image Count |
|---|---|---:|
| Disturbed | Disturbed Galaxies | 1,081 |
| Merging | Merging Galaxies | 1,853 |
| Elliptical | Round Smooth, In-between Round Smooth, Cigar-shaped Smooth | 5,006 |
| Barred Spiral | Barred Spiral | 2,043 |
| Unbarred Spiral | Tight Spiral, Loose Spiral | 4,457 |
| Edge-on | Edge-on without Bulge, Edge-on with Bulge | 3,296 |

The revised taxonomy groups related visual morphologies while retaining distinctions between major galaxy structures.

## 3. Model Architecture

The experiment used **EfficientNetB0** with ImageNet-pretrained weights as the feature extraction backbone.

The classification head consisted of:

- Global Average Pooling
- Dense layer (256 units, ReLU activation)
- Dropout (0.4)
- Output layer (6 units, Softmax activation)

The architecture was kept consistent across training phases, with the output layer adapted for six classes.

## 4. Training Strategy

Training was conducted in three progressive stages.

### Phase 1 — Feature Extraction (Epochs 1–15)

- The pretrained EfficientNetB0 backbone was frozen.
- The newly added classification head was trained.

### Phase 2 — Fine-tuning (Epochs 16–37)

- The top 40 layers of the backbone were unfrozen.
- Fine-tuning was conducted across two consecutive training runs.
- A lower learning rate was used to adapt pretrained features to the galaxy dataset.

### Phase 3 — Fine-tuning with Data Augmentation (Epochs 38–47)

- Data augmentation was introduced during fine-tuning.
- The model was further trained to improve its ability to handle variations in galaxy images.

## 5. Validation Results

The following table presents the final validation metrics from each training stage.

| Training Stage | Validation Accuracy | Validation Loss |
|---|---:|---:|
| Phase 1 | 72.89% | 0.7588 |
| Phase 2 — Run 1 | 77.54% | 0.6544 |
| Phase 2 — Run 2 | 78.49% | 0.6152 |
| Phase 3 — Augmented | **78.69%** | **0.6096** |

The final augmented training phase achieved the highest final validation accuracy and lowest validation loss among the four stages.

These values represent validation performance and should not be interpreted as independent test-set performance.

## 6. Final Model

The Phase 3 checkpoint was selected as the final model based on its validation performance.

- **Model architecture:** EfficientNetB0
- **Output classes:** 6
- **Model file:** `galaxy10_efficientnetb0_phase3.keras`
- **Validation accuracy:** 78.69%
- **Validation loss:** 0.6096

The selected model is stored in the `model-2/` directory.

## 7. Model Evaluation and Interpretability

The experiment included visual evaluation of classification performance across the different training stages.

The evaluation visualisations include:

- Confusion matrices for each training stage.
- Training and validation accuracy curves.
- Training and validation loss curves.
- A consolidated training performance plot covering all 47 epochs.

Grad-CAM was also used to examine the image regions influencing the model's predictions, providing a qualitative view of its classification behaviour.

## 8. Training Visualisations

The `visuals2/` directory contains the visualisations generated during the six-class experiment.

### Master Training Performance

![Master Training Performance](visuals2/master_training_performance.png)

### Individual Training Performance

| Training Stage | Visualisation |
|---|---|
| Phase 1 | `visuals2/phase1_performance.png` |
| Phase 2 — Run 1 | `visuals2/phase2_run1_performance.png` |
| Phase 2 — Run 2 | `visuals2/phase2_run2_performance.png` |
| Phase 3 — Augmented | `visuals2/phase3_augmented_performance.png` |

### Confusion Matrices

| Training Stage | Visualisation |
|---|---|
| Phase 1 | `visuals2/cm_phase1.png` |
| Phase 2 — Run 1 | `visuals2/cm_phase2_run1.png` |
| Phase 2 — Run 2 | `visuals2/cm_phase2_run2.png` |
| Phase 3 | `visuals2/cm_phase3.png` |

## 9. Repository Structure

```text
Galaxy-Morphology/
│
├── comparison_visual/
│
├── galaxy_classifier_app/
│
├── model-2/
│   └── Updated six-class experiment model
│
├── model/
│   └── Original model files
│
├── notebook/
│   └── Experiment notebooks
│
├── visuals/
│   └── Original experiment visualisations
│
├── visuals2/
│   ├── cm_phase1.png
│   ├── cm_phase2_run1.png
│   ├── cm_phase2_run2.png
│   ├── cm_phase3.png
│   ├── master_training_performance.png
│   ├── phase1_performance.png
│   ├── phase2_run1_performance.png
│   ├── phase2_run2_performance.png
│   └── phase3_augmented_performance.png
│
├── .gitignore
└── README.md
```

## 10. Key Takeaways

- Explored a revised six-class morphological taxonomy based on feedback on the original classification task.
- Implemented transfer learning and progressive fine-tuning using EfficientNetB0.
- Investigated the effect of data augmentation during fine-tuning.
- Achieved 78.69% validation accuracy with the final model.
- Used confusion matrices and training visualisations to examine classification performance.
- Applied Grad-CAM to investigate the visual features influencing model predictions.

This experiment demonstrates an application of deep learning to astronomical image analysis and provides a foundation for exploring more physically informative galaxy classification approaches.
