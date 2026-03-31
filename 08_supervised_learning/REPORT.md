# Report: Exercise 10

## Objective
Train and evaluate an EEGNet-based classifier for motor imagery decoding on the BCI IV-2a dataset.

## Method Summary
Two solution variants are included:
- MATLAB pipeline (`exercise10.m`)
- Python/Keras pipeline (`exercise10.py`)

Common workflow:
- load single-trial EEG and labels,
- split into training/validation/test sets,
- build EEGNet architecture,
- optimize with mini-batch training and checkpointing,
- evaluate with accuracy and confusion matrix,
- inspect learned spatial filters.

## Current Run Notes
- The repository contains full source utilities and dataset files for Exercise 10.
- During automated batch figure export in this environment, no Exercise 10 figures were saved to `figures/` (manifest is empty).
- This report is therefore based on the provided solution scripts and project structure.

## Conclusion
Exercise 10 establishes a full supervised deep-learning workflow for EEG classification, from data handling to model interpretation via spatial filter visualization.

