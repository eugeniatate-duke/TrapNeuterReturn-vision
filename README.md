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

Transfer learning using MobileNetV3 with ImageNet-pretrained weights. The final model achieved 88.0% accuracy on the held-out test set.

---
## Robustness Experiment

The final MobileNetV3 model was evaluated under realistic trap-monitoring conditions.

| Condition | Accuracy |
|------------|-----------:|
| Original | 88.0% |
| Reduced Brightness | 87.2% |
| Blurred Images | 64.5% |

Results indicate that the model is highly robust to low-light conditions but significantly affected by image blur, suggesting that motion blur is a more important deployment consideration than darkness.

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

## Dataset

TNRVision uses a custom multi-class image dataset representing species commonly encountered during Trap-Neuter-Return (TNR) operations.

Classes:

| Class | Images |
|---------|---------:|
| Cat | 501 |
| Dog | 501 |
| Raccoon | 260 |
| Opossum | 330 |

Total Images: 1,592

The dataset was curated from multiple publicly available image sources and organized into a unified classification dataset for model development and evaluation.

https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals?select=mammals
https://www.kaggle.com/datasets/debasisdotcom/racoon-detection 

--- 
## Dataset Preparation

Expected raw data structure:

data/raw/
├── cat/
├── dog/
├── raccoon/
└── opossum/

Generate train/validation/test splits:

```bash
python scripts/create_dataset.py
```

The script creates: 

```bash
data/processed/
├── train/
├── val/
└── test/
```

using a 70/15/15 split.
---

## Setup

Create and activate a virtual environment:

```bash

python -m venv .venv

source .venv/bin/activate
```
Install dependencies: 
```bash
pip install -r requirements.txt
```
Generate train/validation/test splits:
```bash
python scripts/create_dataset.py
```
Train models:
```bash:
python scripts/train_baseline.py
python scripts/train_classical.py
python scripts/train_deep.py
```

---
## Results

| Model | Accuracy |

|---------|---------:|

| Majority-Class Baseline | 31.4% |

| HOG + SVM | 51.1% |

| MobileNetV3 | 88.0% |

The MobileNetV3 transfer learning model achieved the strongest performance and was selected as the deployment model for the TNRVision application.

--- 

## Deployment

TNRVision is deployed as a Hugging Face Space using Streamlit.

The application performs inference using the trained MobileNetV3 model and provides:
- Animal classification
- Confidence scores
- TNR-specific recommendations
- Human-review alerts for low-confidence predictions

---

## Live Demo

Hugging Face Space: https://huggingface.co/spaces/Eugenia-Tate-Duke/tnrvision

### Key Features

- Four-class animal classification

- AI-assisted recommendation engine

- Human-review workflow for low-confidence predictions

- Robustness evaluation under realistic monitoring conditions

- Publicly deployed Streamlit application
--- 

## License

Academic use only.
