# Results and technical notes

_To be completed after the final training run. Numbers come from `reports/comparison.md` and `reports/<model>/metrics.json`._

## 1. Model comparison
- Table: baseline vs ResNet18 (test accuracy, macro P/R/F1, ECE, size, latency, training time)
- Which model was selected and why (validation macro-F1), and what transfer learning added

## 2. Error analysis
- Confusion matrix: which class pairs get confused
- Misclassified examples (`reports/<model>/misclassified.png`): likely causes
- Least-confident correct predictions: borderline cases
- Class-level weaknesses (per-class F1)

## 3. False positives and false negatives in this setting
- What a missed defect (FN) vs a false alarm (FP) costs on a production line
- How the low-confidence threshold trades one for the other

## 4. Data leakage: what we did
- Test = Kaggle validation folder, never used for training, early stopping, calibration or model choice
- MD5 duplicate check; duplicates kept in the same split

## 5. Limitations
- Small, clean, balanced, lab-style dataset: 200×200 crops, one defect per image
- Real lines: different cameras, lighting, resolution, multiple or no defects, "no defect" class missing
- The model always outputs one of 6 classes, even for unrelated images (the validation only warns)

## 6. Next improvements
- Add a "no defect / unknown" class or an out-of-distribution detector
- Cross-validation for more reliable estimates on a small dataset
- Test on images from another source (domain shift)
- Lighter model (MobileNetV3) for edge deployment; ONNX export
