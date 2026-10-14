# Results and technical notes

All numbers are from one full `dvc repro` run (seed 42, Apple M-series GPU via MPS). Sources: `reports/comparison.md`, `reports/<model>/metrics.json`, `models/<model>/train_summary.json`.

## 1. Model comparison

| | Baseline CNN (from scratch) | ResNet18 (ImageNet, fine-tuned) |
|---|---|---|
| Val accuracy / macro-F1 | 99.31% / 99.31% | **100% / 100%** |
| **Test accuracy** (n = 360) | 99.44% (2 errors) | **100% (0 errors)** |
| Test macro precision / recall / F1 | 99.45% / 99.44% / 99.44% | 100% / 100% / 100% |
| Test ECE: before → after temperature scaling | 0.0205 → **0.0045** (T = 0.65) | 0.0016 (T = 1.0, not fitted: 0 val errors) |
| Parameters / file size | 1.17 M / 4.7 MB | 11.18 M / 44.8 MB |
| CPU latency (single image, incl. preprocessing) | 11.7 ms | **7.8 ms** |
| Training time / best epoch | 567 s / epoch 26 of 30 | **258 s** / epoch 16 of 20 |

**Selected model: ResNet18**, chosen on validation macro-F1 (100% vs 99.31%). The test set played no part in the choice.

### What transfer learning actually added
The baseline is already at 99.4%, so the honest conclusion is that **NEU is an easy dataset**: lab-quality 200×200 crops with one centred defect type per image. The gain from ResNet18 is real but small in accuracy. It shows up more clearly in three other places:
1. **Faster, more stable training.** ResNet18 passed 99% val accuracy one epoch into fine-tuning (epoch 6) and finished in 258 s. The baseline needed about 22 epochs and 567 s to get there, and its val loss jumped around for the first ~15 epochs (see `reports/baseline/training_curves.png`).
2. **Robustness on the hard class.** Both baseline errors involve *inclusion*, and ResNet18 gets every inclusion right.
3. **Faster on CPU despite 10× more parameters.** Parameter count is not compute. ResNet18 shrinks the image to 56×56 in its first two layers (stride-2 conv + max-pool). The baseline runs two convolutions at the full 224×224 before its first pooling, which costs more operations per image.

The trade-off is file size: 44.8 MB vs 4.7 MB. That's fine for a server or Streamlit Cloud. For an edge camera, the baseline or a MobileNet would be worth considering.

## 2. Error analysis

### Baseline: 2 test errors (`reports/baseline/misclassified.png`)
| Image | True → predicted | Confidence | Likely cause |
|---|---|---|---|
| `inclusion_283.jpg` | inclusion → **scratches** | 96% | A single long, dark vertical streak. Elongated inclusions and scratches share the same shape (a thin line), so the baseline relied on shape rather than texture. This is a **confident** mistake, the most dangerous kind: the low-confidence warning would not catch it. |
| `pitted_surface_282.jpg` | pitted_surface → **inclusion** | 60.3% | Scattered dark spots on a bright background. Small, isolated pits look like the dark specks inside inclusions. The model was unsure, and the confidence sits right at the app's 60% warning threshold (just above it, so no warning would show). |

### ResNet18: 0 test errors, so we looked at its least confident correct predictions (`reports/resnet18/least_confident_correct.png`)
Of its 12 least confident correct predictions, **10 are inclusions** (lowest: 79%, 88%, 91%, 92%), and they are all vertical dark streaks. The other two are a patches image and a scratches image, both at 99–100%. Both models therefore point to the **same weak spot: inclusion vs scratches.** If more errors appear on new data, they will most likely be here.

### Per-class view
- Baseline: crazing, patches and rolled-in_scale are perfect. Inclusion has the lowest F1 (0.983). Scratches loses some precision (0.984) because it absorbed one inclusion.
- ResNet18: all six classes have F1 = 1.000 on this test set.

## 3. How certain is "100%"?
360 test images with 0 errors does **not** mean the model will never be wrong. By the "rule of three", with 0 errors in n trials the 95% upper bound on the true error rate is about 3/n = **0.83%**. So the honest claim is that the true accuracy on images like these is **very likely above ~99.2%**. A bigger or harder test set would be needed to tell 99.5% from 100%.

## 4. Confidence calibration
- **Baseline:** temperature T = 0.65 (< 1), meaning the model was **under-confident**. Dropout and augmentation during training make it hedge. Fitted on val, the temperature cut test ECE from 0.0205 to 0.0045 (about 4.6× better) without changing a single prediction.
- **ResNet18:** zero validation errors, so there is nothing to calibrate against. Minimising NLL would push T toward 0 and make the model absurdly over-confident. We keep T = 1.0 and log the reason in `models/resnet18/calibration.json`. Its uncalibrated test ECE is already 0.0016.

## 5. False positives and false negatives on a production line
In a 6-class setting, every error is both a false negative for the true class and a false positive for the predicted class. Example: the baseline's inclusion → scratches error is an FN for *inclusion* and an FP for *scratches*. What it costs depends on the action each class triggers:
- If inclusions mean scrapping the coil but scratches only mean polishing, then **inclusion → scratches is the expensive error**: a defective coil ships.
- The low-confidence threshold (default 60%) sends uncertain images to a human. Raising it catches more borderline cases (like the 60.3% pitted-surface error) at the cost of more manual checks. It cannot catch confident errors (like the 96% one), which is why per-class costs and periodic human audits still matter.

## 6. Data leakage: what we did
- **Test = Kaggle `validation/`**, used only by `src/evaluate.py`. Early stopping, temperature fitting and model selection all use our val split, which was carved from Kaggle `train/`.
- **MD5 check:** one exact duplicate pair inside train (`patches_101` = `patches_105`), kept in the same split. No image appears in both train and test.
- **Same preprocessing everywhere:** evaluation, the CLI and the app all load the transform from the saved `preprocess.json`, so there is no train/serve skew.
- **Remaining risk:** near-duplicates. NEU images may be crops from the same steel strips, and MD5 only catches byte-identical files. A perceptual-hash check would be the next step.

## 7. Limitations
- **Easy, clean data:** one defect type per image, centred, same camera and lighting, balanced classes. Real inspection images vary in camera, lighting, scale and resolution.
- **No "no defect" class:** the model always answers one of 6 classes, even for a clean surface or an unrelated photo. The app only *warns* on colour or flat images; it cannot truly reject them.
- **Small test set** (360): see the rule-of-three bound above.
- **Single run:** one seed and one split. Cross-validation or several seeds would give error bars.

## 8. Next improvements
1. Add a "no defect / unknown" class, or an out-of-distribution score (e.g. max-softmax or energy threshold), so the app can say "this is not a steel surface".
2. Test on images from a different source (another dataset or camera) to measure domain shift, which is the real risk in deployment.
3. Run 5-fold cross-validation and several seeds to report mean ± std.
4. Train with harder inclusion vs scratches examples (targeted augmentation, or focal loss).
5. For edge devices: MobileNetV3 or ONNX/quantised ResNet18, re-checking accuracy and latency.
