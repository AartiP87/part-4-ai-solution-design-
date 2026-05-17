# Part 4: AI Solution Design — Healthcare Medical Image Triage

## Overview

This project presents an AI-based solution design for **automated medical image triage in healthcare**, focusing on chest X-ray classification using deep learning (CNN / Transfer Learning). The goal is to assist radiologists by flagging abnormal scans, reducing manual review time, and improving patient outcomes.

---

## Repository Structure

```
part-4-ai-solution-design/
│
├── README.md                        ← This file
├── solution_report.md               ← Full written report (Tasks 1–8)
├── diagrams/
│   └── solution_architecture.png    ← System architecture diagram
└── notebook/
    ├── ai_solution_design.ipynb     ← Jupyter notebook with analysis & visuals
    └── requirements.txt             ← Python dependencies
```

---

## Data Sources

| Dataset | Description | Link |
|---|---|---|
| **NIH Chest X-ray Dataset** | 112,000+ frontal chest X-ray images with 14 disease labels | [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **CheXpert Dataset** | 224,316 chest radiographs from Stanford Medicine | [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **MIMIC-CXR** | 227,827 imaging studies from Beth Israel Deaconess Medical Center | [https://physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/) |
| **Reference Catalog (Assignment)** | AI use-case reference data provided for this assignment | [Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing) |

---

## Domain

**Healthcare** — Medical Imaging / Radiology Triage

## AI Task Type

**Image Classification** using Convolutional Neural Networks (CNN) with Transfer Learning

## Model Recommended

**DenseNet-121** (pre-trained on ImageNet, fine-tuned on chest X-rays)

---

## Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd part-4-ai-solution-design

# Install dependencies
pip install -r notebook/requirements.txt

# Launch the notebook
jupyter notebook notebook/ai_solution_design.ipynb
```

---

## Key Business Impact

| Metric | Before AI | After AI (Projected) |
|---|---|---|
| Manual review hours/month | ~490 hrs | ~160 hrs |
| Average resolution time | ~30 hrs | ~10 hrs |
| Error rate | ~7% | ~2% |
| Patient satisfaction | 6.7 / 10 | 8.2 / 10 |

---

## Responsible AI

This solution is designed as a **decision-support tool**, not a replacement for radiologists. All predictions are reviewed by a licensed medical professional before any clinical action is taken.

---

## Author

AI Business Analyst — Part 4 Assignment Submission
