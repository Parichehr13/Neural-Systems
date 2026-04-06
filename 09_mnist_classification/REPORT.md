# MNIST Classification Report

## Objective
Train and evaluate a supervised MNIST digit classifier, then document model behavior using terminal metrics and all exported figures from `mnist_classification.py`.

## Run Configuration
- Dataset: `mnist`
- Model type: `dense`
- Input: normalized grayscale images, shape `(28, 28, 1)`
- Output: 10 classes (`0-9`)
- Optimizer: SGD (`learning_rate=0.001`, `momentum=0.9`)
- Loss: categorical cross-entropy
- Batch size: `128`
- Run setting used for this report: `EX11_MAX_EPOCHS=5`

## Model Layers (Terminal Summary)
| Layer | Output Shape | Params |
|---|---|---:|
| `flatten (Flatten)` | `(None, 784)` | 0 |
| `dense (Dense)` | `(None, 128)` | 100,480 |
| `dropout (Dropout)` | `(None, 128)` | 0 |
| `dense_1 (Dense)` | `(None, 64)` | 8,256 |
| `dropout_1 (Dropout)` | `(None, 64)` | 0 |
| `dense_2 (Dense)` | `(None, 10)` | 650 |

Total params: `109,386`

## Performance From Terminal Output
| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Training | 0.9045 | 0.9029 | 0.9041 |
| Validation | 0.9042 | 0.9030 | 0.9040 |
| Test | 0.9096 | 0.9080 | 0.9092 |

Custom digit inference from terminal:
- Predicted class: `1`
- Class probabilities: `[0.0034, 0.2974, 0.0243, 0.0349, 0.1009, 0.0767, 0.0621, 0.0331, 0.1672, 0.1999]`

## Figures From `mnist_classification.py`

### Figure Index (Direct Links)
- [Figure 1 file](./figures/mnist_classification_fig_001_sample_input.png)
- [Figure 2 file](./figures/mnist_classification_fig_002_training_history.png)
- [Figure 3 file](./figures/mnist_classification_fig_003_validation_confusion_raw.png)
- [Figure 4 file](./figures/mnist_classification_fig_004_validation_confusion_norm.png)
- [Figure 5 file](./figures/mnist_classification_fig_005_test_confusion_raw.png)
- [Figure 6 file](./figures/mnist_classification_fig_006_test_confusion_norm.png)
- [Figure 7 file](./figures/mnist_classification_fig_007_misclassified_examples.png)
- [Figure 8 file](./figures/mnist_classification_fig_008_custom_digit.png)

### Figure 1 - Sample Input
Shows one MNIST training image used as an example input.

![Figure 1 - Sample Input](./figures/mnist_classification_fig_001_sample_input.png)

### Figure 2 - Training History
Training and validation loss/accuracy across epochs.

![Figure 2 - Training History](./figures/mnist_classification_fig_002_training_history.png)

### Figure 3 - Validation Confusion Matrix (Raw)
Absolute misclassification counts on the validation split.

![Figure 3 - Validation Confusion Raw](./figures/mnist_classification_fig_003_validation_confusion_raw.png)

### Figure 4 - Validation Confusion Matrix (Normalized)
Per-class normalized validation confusion matrix.

![Figure 4 - Validation Confusion Normalized](./figures/mnist_classification_fig_004_validation_confusion_norm.png)

### Figure 5 - Test Confusion Matrix (Raw)
Absolute misclassification counts on the test split.

![Figure 5 - Test Confusion Raw](./figures/mnist_classification_fig_005_test_confusion_raw.png)

### Figure 6 - Test Confusion Matrix (Normalized)
Per-class normalized test confusion matrix.

![Figure 6 - Test Confusion Normalized](./figures/mnist_classification_fig_006_test_confusion_norm.png)

### Figure 7 - Misclassified Examples
Examples where predicted labels differ from true labels.

![Figure 7 - Misclassified Examples](./figures/mnist_classification_fig_007_misclassified_examples.png)

### Figure 8 - Custom Digit
User-provided digit image used for inference.

![Figure 8 - Custom Digit](./figures/mnist_classification_fig_008_custom_digit.png)

## Notes
- Terminal logs used for this report are saved in `09_mnist_classification/terminal_output.txt`.
- Figure export list is tracked in `09_mnist_classification/figures/mnist_classification_manifest.json`.

