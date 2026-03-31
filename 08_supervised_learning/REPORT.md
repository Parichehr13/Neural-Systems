# Supervised Learning Report

## Objective
Train and evaluate an EEGNet-based classifier for motor imagery decoding on the BCI IV-2a dataset.

## Method Summary
Two solution variants are included:
- MATLAB pipeline (`supervised_learning.m`)
- Python/Keras pipeline (`supervised_learning.py`)

Common workflow:
- load single-trial EEG and labels,
- split into training/validation/test sets,
- build EEGNet architecture,
- optimize with mini-batch training and checkpointing,
- evaluate with accuracy and confusion matrix,
- inspect learned spatial filters.

## Current Run Notes
- The repository contains full source utilities and dataset files for this module.
- During automated batch figure export in this environment, no module figures were saved to `figures/` (manifest is empty).
- This report is therefore based on the provided solution scripts and project structure.

## Conclusion
This module establishes a full supervised deep-learning workflow for EEG classification, from data handling to model interpretation via spatial filter visualization.


