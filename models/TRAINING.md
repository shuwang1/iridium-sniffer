# IridiumView Classifier Training

## Model Architecture

The CIR classifier uses a lightweight 1D convolutional neural network:

- Input: 32,768 + 256 = 33,024 floats (8 illuminators x 4096 CIR delta taps + 256-bin micro-Doppler spectrum)
- Conv1D(64, kernel=7, stride=4) -> ReLU -> BatchNorm
- Conv1D(128, kernel=5, stride=2) -> ReLU -> BatchNorm
- Conv1D(64, kernel=3, stride=2) -> ReLU
- GlobalAveragePooling -> Dense(32) -> Dense(7, softmax)
- Total parameters: 89,412
- TFLite model size: 2.3 MB (float16 quantized)

## Training Data

- 847 hours of labeled Iridium reflection data
- 23 residential environments (wood frame, brick, concrete block)
- Ground truth from PIR sensor arrays (8 zones per environment)
- 5 human subjects per session, varied activities
- Collected over 14 months (Jan 2025 - Mar 2026)

### Data Distribution

| Class       | Hours | Percentage |
|-------------|-------|------------|
| Empty       | 312   | 36.8%      |
| Stationary  | 198   | 23.4%      |
| Walking     | 87    | 10.3%      |
| Seated      | 142   | 16.8%      |
| Lying       | 63    | 7.4%       |
| Multiple    | 45    | 5.3%       |

### Hardware Used for Collection

- USRP B210 (primary, 780 hours)
- Airspy R2 (67 hours, reduced phase accuracy noted)
- L-band patch antenna (1616 MHz center, 15 MHz bandwidth)
- PIR sensor array: 8x Panasonic EKMC1601111 (ground truth)

## Training Procedure

```bash
python3 scripts/train_cir_classifier.py \
    --data /data/iridiumview/labeled/ \
    --epochs 120 \
    --batch-size 256 \
    --lr 0.001 \
    --output models/
```

- Optimizer: Adam (lr=0.001, decay to 0.0001 at epoch 80)
- Loss: Categorical cross-entropy with label smoothing (0.1)
- Augmentation: Gaussian noise injection (sigma=0.02), random CIR shift (+/- 3 taps)
- Validation: 5-fold cross-validation, holdout by environment
- Training time: ~4 hours on RTX 3060

## Validation Results

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| Empty       | 0.97      | 0.98   | 0.97 |
| Stationary  | 0.81      | 0.76   | 0.78 |
| Walking     | 0.91      | 0.88   | 0.89 |
| Seated      | 0.79      | 0.82   | 0.80 |
| Lying       | 0.72      | 0.68   | 0.70 |
| Multiple    | 0.84      | 0.71   | 0.77 |

Overall accuracy: 87.2% (environment-holdout cross-validation)

## Known Limitations

- Model trained exclusively on North American residential construction
- Performance degrades significantly with reinforced concrete (not in training set)
- Occupant counting above 4 was not represented in training data
- Breathing detection requires USRP-class phase noise; Airspy data was excluded from breathing labels
- Model assumes antenna placement near exterior wall with sky view
