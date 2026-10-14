# Model comparison

Selected for deployment (by **validation** macro-F1): **resnet18**

| Metric | baseline | resnet18 |
|---|---|---|
| Val accuracy | 99.31% | 100.00% |
| Val macro-F1 | 99.31% | 100.00% |
| **Test accuracy** | 99.44% | 100.00% |
| Test macro precision | 99.45% | 100.00% |
| Test macro recall | 99.44% | 100.00% |
| **Test macro-F1** | 99.44% | 100.00% |
| Test errors (of 360) | 2 | 0 |
| Test ECE (calibrated) | 0.0045 | 0.0016 |
| Parameters | 1.17 M | 11.18 M |
| Model size (MB) | 4.72 | 44.79 |
| CPU latency (ms/image) | 11.73 | 7.84 |
| Training time (s) | 567.0 | 258.3 |
| Best epoch / epochs run | 26 / 30 | 16 / 20 |
