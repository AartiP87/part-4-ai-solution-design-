# AI Solution Design Report — Healthcare Medical Image Triage

**Domain:** Healthcare  
**AI Task Type:** Image Classification  
**Model:** CNN with Transfer Learning (DenseNet-121)  
**Date:** 2025

---

## Task 1: Business Domain

**Selected Domain: Healthcare**

Healthcare was selected because it represents one of the highest-impact areas for AI deployment. Medical imaging is a data-intensive, time-critical process where AI can meaningfully reduce diagnostic delays, support overloaded radiology departments, and ultimately improve patient outcomes. The reference catalog confirms this domain maps directly to Image Classification with CNN or transfer learning models, with recall and sensitivity as primary evaluation metrics.

---

## Task 2: Business Problem Definition

### Problem Statement

Hospitals and diagnostic centres process thousands of chest X-rays daily to screen for conditions such as pneumonia, pleural effusion, cardiomegaly, and pulmonary nodules. Radiologists must manually review every scan — a process that is slow, expensive, and subject to human fatigue-related errors.

### Who Are the Stakeholders?

| Stakeholder | Role |
|---|---|
| Radiologists | Primary users — receive AI-flagged cases for priority review |
| Emergency Physicians | Need fast triage decisions for critical patients |
| Hospital Administrators | Concerned with throughput, cost, and liability |
| Patients | Benefit from faster, more accurate diagnoses |
| Compliance/Legal Teams | Ensure AI use meets medical device regulations |

### Current Manual Process

1. A physician orders a chest X-ray for a patient.
2. The scan is uploaded to the hospital's PACS (Picture Archiving and Communication System).
3. A radiologist is assigned the case and manually reviews it — often hours or days later depending on queue length.
4. A written radiology report is issued and sent back to the ordering physician.
5. Clinical decisions are made based on that report.

### Limitations of the Current Process

- **Long turnaround times:** Average resolution time of 30+ hours based on reference KPI data.
- **High manual workload:** Radiologists review 490+ hours of cases per month — leading to fatigue and burnout.
- **Error rate of ~7%:** Missed findings or delayed reporting occur, especially under high volume.
- **No automated prioritisation:** Critical scans (e.g., tension pneumothorax) can sit in the same queue as routine follow-ups.
- **Scalability constraints:** Hiring more radiologists is expensive and slow; AI can scale instantly.

---

## Task 3: AI Task Type

**Selected Task Type: Image Classification**

### Justification

The input data is chest X-ray images (unstructured, pixel-based). The output required is a label (e.g., "Normal", "Pneumonia", "Effusion", "Nodule", etc.). This maps directly to a multi-label image classification problem.

- Each image needs to be assigned one or more diagnostic labels.
- The model must learn spatial features (textures, densities, shapes) within the image — which is the core strength of convolutional neural networks.
- Transfer learning from pre-trained models (trained on ImageNet) accelerates training and improves performance even with limited labelled medical data.

This is **not** a regression task (no continuous numeric output), **not** object detection (we don't need bounding boxes at this stage), and **not** NLP (the data is images, not text).

---

## Task 4: Data Requirement Plan

### Type of Data Needed

| Data Element | Type | Format |
|---|---|---|
| Chest X-ray images | Unstructured | DICOM / PNG / JPEG |
| Diagnostic labels | Structured | CSV / JSON (multi-label) |
| Patient metadata | Structured | Age, sex, view position |
| Radiologist reports | Unstructured (text) | PDF / HL7 |

### Structured vs. Unstructured

- **Primary input:** Unstructured (X-ray images)
- **Labels/metadata:** Structured

### Input Features

- Raw pixel values of the chest X-ray (resized to 224×224 for DenseNet-121 input)
- Image view position: PA (posteroanterior) or AP (anteroposterior)
- Patient age and sex (optional metadata input for multi-modal extensions)

### Target Variable / Labels

Multi-label binary classification across 14 pathology classes (following NIH convention):

> Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia

Each image receives a binary label (0/1) per class. "No Finding" is also a valid label.

### Data Collection Method

| Source | Description |
|---|---|
| NIH Chest X-ray Dataset | 112,120 images, 14 labels — publicly available |
| CheXpert (Stanford) | 224,316 images — uncertainty labels handled via label-smoothing |
| MIMIC-CXR (PhysioNet) | 227,827 studies with paired radiology reports |
| Internal hospital PACS | De-identified scans for domain adaptation (future phase) |

### Data Quality Risks

| Risk | Description | Mitigation |
|---|---|---|
| Label noise | NIH labels were NLP-extracted from reports — not hand-labelled | Use CheXpert for validation; apply label smoothing |
| Class imbalance | "No Finding" dominates; rare conditions underrepresented | Weighted loss functions, oversampling, augmentation |
| Image quality variation | Different scanners, voltages, resolutions | Normalisation, histogram equalisation, CLAHE |
| Patient privacy | X-rays may contain identifiable metadata in DICOM headers | Strip DICOM metadata; apply de-identification pipeline |
| Distribution shift | Model trained on US hospital data may not generalise globally | Collect diverse training data; test on local hospital samples |

---

## Task 5: Model Recommendation

### Recommended Architecture: DenseNet-121 (Transfer Learning)

**DenseNet-121** (Densely Connected Convolutional Network) is the recommended architecture, pre-trained on ImageNet and fine-tuned on chest X-ray data. This is the same architecture used in CheXNet (Rajpurkar et al., Stanford, 2017), which matched radiologist-level performance on pneumonia detection.

### Architecture Overview

```
Input (224×224×3 RGB X-ray)
        ↓
DenseNet-121 Backbone (pre-trained on ImageNet)
[Dense Block 1] → [Transition Layer] → [Dense Block 2] → ...
        ↓
Global Average Pooling
        ↓
Fully Connected Layer (1024 → 512)
        ↓
Output Layer (14 sigmoid activations — one per pathology class)
```

### Why DenseNet-121?

| Reason | Explanation |
|---|---|
| Dense connectivity | Each layer receives feature maps from all preceding layers — reduces vanishing gradient, improves feature reuse |
| Transfer learning | Pre-trained weights provide rich low-level feature detectors (edges, textures) that transfer well to medical images |
| Proven in medical imaging | CheXNet achieved AUC > 0.85 on most pathologies using this architecture |
| Compact and efficient | Fewer parameters than VGG or ResNet-152 — suitable for hospital deployment |
| Multi-label support | Sigmoid output (not softmax) supports simultaneous detection of multiple conditions |

### Alternative Models Considered

| Model | Verdict |
|---|---|
| ResNet-50 | Good baseline — slightly lower performance than DenseNet on this task |
| EfficientNet-B4 | Strong performance but harder to interpret |
| Vision Transformer (ViT) | Excellent but requires more data and compute |
| Basic CNN from scratch | Insufficient — too little labelled medical data |

### Training Strategy

- **Phase 1:** Freeze backbone, train only the classification head (5 epochs)
- **Phase 2:** Unfreeze all layers, fine-tune with low learning rate (1e-4)
- **Loss function:** Binary Cross-Entropy with class weights (to address imbalance)
- **Optimiser:** Adam with cosine learning rate schedule
- **Data augmentation:** Random horizontal flip, rotation (±10°), brightness/contrast jitter, zoom

---

## Task 6: Evaluation Plan

### Technical Metrics

| Metric | Purpose | Target |
|---|---|---|
| AUC-ROC (per class) | Measures discrimination ability at all thresholds | > 0.85 per class |
| Sensitivity / Recall | Fraction of true positives caught — critical in medical context | > 0.90 for critical conditions |
| Specificity | Fraction of true negatives — avoids unnecessary referrals | > 0.85 |
| F1-Score | Balance of precision and recall | > 0.80 |
| Precision | Positive predictive value | > 0.80 |

> **Note:** In medical settings, **recall (sensitivity) is prioritised over precision** — it is worse to miss a diagnosis than to flag an unnecessary review.

### Business Metrics

| KPI | Baseline (from reference data) | Target after AI |
|---|---|---|
| Manual processing hours/month | ~490 hrs | < 160 hrs (67% reduction) |
| Average resolution time | ~30 hrs | < 10 hrs |
| Error rate (missed findings) | ~7% | < 2% |
| Patient satisfaction score | 6.7 / 10 | > 8.0 / 10 |
| Monthly case throughput | ~2,800 cases | > 4,500 cases |

### Possible Failure Cases

1. **False negatives on rare conditions:** The model misses a nodule or early-stage mass → patient goes undiagnosed.
2. **False positives causing alarm:** Model flags normal scans as abnormal → unnecessary patient anxiety and follow-up costs.
3. **Performance degradation on unseen scanner types:** Hospital acquires a new CT/X-ray machine with different image characteristics.
4. **Label drift:** New disease patterns (e.g., post-COVID fibrosis) not represented in training data.

### Human Review / Validation Process

- All AI outputs are reviewed by a licensed radiologist before clinical use.
- A confidence threshold is applied: cases below 70% model confidence are automatically escalated.
- A monthly audit compares AI predictions to final radiologist reports to track drift.
- A feedback loop allows radiologists to flag incorrect predictions, which feeds retraining.

---

## Task 7: Responsible AI Considerations

### 1. Bias in Data

**Risk:** Training data (NIH, CheXpert) is predominantly from US hospitals — the model may perform poorly on patients from other ethnicities, age groups, or geographies.  
**Mitigation:** Evaluate performance separately by age, sex, and scanner type. Collect diverse local data for fine-tuning before clinical deployment.

### 2. Incorrect Predictions

**Risk:** A missed pneumothorax or undetected lung mass could have life-threatening consequences.  
**Mitigation:** Position AI as a triage support tool only. All critical flags require radiologist sign-off. Set recall thresholds conservatively (prefer false positives over false negatives for serious conditions).

### 3. Privacy Concerns

**Risk:** Chest X-rays and associated metadata are sensitive personal health information protected under HIPAA, GDPR, and local data protection laws.  
**Mitigation:** Strip DICOM metadata before training. Store model weights separately from patient data. Implement role-based access control. Use federated learning for multi-hospital training where possible.

### 4. Over-reliance on AI

**Risk:** Radiologists may "rubber stamp" AI outputs without independent review — especially under time pressure.  
**Mitigation:** Design the UI to show AI confidence scores, not just labels. Require radiologists to confirm or override each prediction. Track override rates and investigate when they drop too low.

### 5. Impact on Healthcare Workers

**Risk:** Fear of AI replacing radiologist jobs may reduce adoption or create adversarial behaviour (e.g., ignoring AI flags deliberately).  
**Mitigation:** Frame AI as a productivity assistant, not a replacement. Communicate that the goal is to handle routine volume so radiologists can focus on complex cases. Include radiologists in the design and validation process.

### 6. Regulatory and Liability Risk

**Risk:** In most countries, AI medical devices require regulatory approval (FDA 510(k), CE Mark, SFDA, etc.) before clinical use.  
**Mitigation:** Engage regulatory affairs team early. Document model training process, validation datasets, and performance benchmarks. Maintain a model card and audit log.

---

## Task 8: Final Solution Summary

### One-Page Solution Summary

---

**Problem:**  
Hospital radiology departments are overwhelmed by the volume of chest X-rays requiring manual review. Current processes result in average resolution times exceeding 30 hours, a ~7% error rate, and radiologist burnout. There is no automated triage to prioritise critical cases.

**Proposed AI Solution:**  
Deploy a deep learning image classification system (DenseNet-121 with transfer learning) that analyses chest X-rays and predicts the likelihood of 14 pathological conditions. The system integrates with the hospital PACS and presents a prioritised worklist to radiologists, with the highest-confidence abnormal scans surfaced first.

**Required Data:**  
- Labelled chest X-ray images (NIH / CheXpert / MIMIC-CXR — ~500,000+ images)
- Patient demographic metadata (age, sex, view position)
- Historical radiologist reports for validation
- Internal hospital PACS data for domain adaptation

**Model Recommendation:**  
DenseNet-121 pre-trained on ImageNet, fine-tuned with binary cross-entropy loss on multi-label chest X-ray data. Trained in two phases: head-only fine-tuning followed by full network fine-tuning. Deployed as a REST API integrated into the hospital PACS workflow.

**Expected Business Impact:**

| KPI | Improvement |
|---|---|
| Manual hours saved | ~330 hrs/month (~67%) |
| Resolution time | From 30 hrs → under 10 hrs |
| Error rate reduction | From 7% → below 2% |
| Patient satisfaction | From 6.7 → above 8.0 |
| Additional cases handled | +60% capacity without new hires |

**Risks and Mitigation Plan:**

| Risk | Mitigation |
|---|---|
| Missed critical diagnoses | Conservative recall threshold; mandatory radiologist sign-off |
| Data bias (demographic) | Evaluate by subgroup; collect diverse training samples |
| Privacy / HIPAA compliance | DICOM de-identification; federated learning; encrypted storage |
| Over-reliance on AI | UI shows confidence scores; override rates monitored |
| Regulatory non-compliance | Pursue FDA/CE approval; maintain model card and audit trail |
| Performance drift | Monthly evaluation vs. radiologist ground truth; retraining pipeline |

---

*This solution is designed as a clinical decision-support system. The AI model assists but never replaces the radiologist. Human oversight is embedded at every step of the workflow.*

---

## References

- Rajpurkar, P. et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* Stanford ML Group.
- Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.
- Irvin, J. et al. (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels.* AAAI.
- Johnson, A. et al. (2019). *MIMIC-CXR: A large publicly available database of labeled chest radiographs.* PhysioNet.
- Huang, G. et al. (2017). *Densely Connected Convolutional Networks.* CVPR.
