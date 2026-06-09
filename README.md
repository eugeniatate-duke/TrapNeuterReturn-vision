# TNRVision

TNRVision is a computer vision decision-support system designed to assist Trap-Neuter-Return (TNR) programs by automatically identifying animals captured by trap-monitoring cameras.

The goal is to help volunteers distinguish target animals (cats) from non-target animals (e.g., dogs, raccoons, opossums) and provide recommendations that reduce unnecessary trap checks and improve operational efficiency.

This project was developed for Project 1 (Computer Vision) in Duke University's AIPI 540: Deep Learning Applications.

---

## Problem Statement

Community cat populations can grow rapidly when unmanaged. Trap-Neuter-Return (TNR) programs help control population growth, but volunteers often spend significant time monitoring traps and reviewing images from camera systems.

TNRVision explores whether computer vision models can automatically classify animals appearing near traps and support volunteers in identifying likely target species.

---

## Project Objectives

- Build and evaluate a naive baseline model
- Build and evaluate a classical computer vision model
- Build and evaluate a deep learning model
- Compare model performance using standard classification metrics
- Conduct robustness experiments under simulated low-light and image degradation conditions
- Deploy a public-facing interactive application for model inference

---

## Animal Classes

The initial version of TNRVision classifies:

- Cat
- Dog
- Raccoon
- Opossum

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── scripts
├── models
├── data
│   ├── raw
│   ├── processed
│   └── outputs
├── notebooks
├── reports
└── artifacts
```

---

## Modeling Approaches

### Naive Baseline

Majority-class classifier.

### Classical Computer Vision

Histogram of Oriented Gradients (HOG) feature extraction combined with a traditional machine learning classifier (e.g., SVM or Random Forest).

### Deep Learning

Transfer learning using a pretrained convolutional neural network (CNN).

---

## Planned Experiment

### Robustness Under Real-World Trap Conditions

The project will evaluate model performance under simulated deployment conditions including:

- Reduced brightness
- Motion blur
- Image blur

The objective is to assess how well different approaches generalize to realistic nighttime monitoring scenarios.

---

## Data

Raw image datasets are not stored in the repository.

Expected structure:

```text
data/raw/
├── cat/
├── dog/
├── raccoon/
└── opossum/
```

Dataset acquisition and preprocessing scripts will be provided in the `scripts/` directory.

---

## Setup

Coming soon.

---

## Deployment

Coming soon.

---

## License

Academic use only.