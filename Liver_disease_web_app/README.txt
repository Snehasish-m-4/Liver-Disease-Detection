
LIVER IMAGE PREDICTIVE MODEL
============================

Architecture:
EfficientNet-B3
+
Channel Attention
+
Spatial Attention
+
Multi-Head Self-Attention
+
Classifier

Classes:
0 - ballooning
1 - fibrosis
2 - inflammation
3 - steatosis

Input:
224 x 224 RGB image

Inference preprocessing:
Resize -> ToTensor -> ImageNet Normalize

Checkpoint:
best_stage2_cutmix_model.pth

Best validation accuracy:
97.75%

Best validation epoch:
134

Total parameters:
21,226,638


PREDICTIVE COMPONENTS
=====================

Model 1:
Basic image classification

Model 2:
Class probability analysis

Model 3:
Grad-CAM explainability


IMPORTANT:
This model is intended for research/educational demonstration.
Its output is a model prediction and not a clinical diagnosis.
