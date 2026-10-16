# AI tools and prompt log

**Tools used:** Claude (Anthropic)

**How I used it:** to plan the approach, draft the code and documentation, and explain design choices. I made the final decisions (listed below), ran all training and evaluation, and reviewed every file before committing.

---

## How I worked with the AI
- **Gave it the real context first:** the assignment PDF and my dataset, so the plan followed the brief, not generic advice.
- **Asked for options, then chose myself:** e.g. baseline type, pretrained model, framework, extras. My scope decisions are in prompt 3.
- **Asked "why" until I understood:** why two models, where the accuracy number comes from, what each pipeline stage does.
- **Pushed back when it went too far:** it built more than I wanted, so I limited it to what the brief asks (prompt 8).
- **Debugged with real error messages:** I pasted the actual Terminal output instead of describing the problem (prompt 5).
- **Verified instead of trusting:** tests, the confusion-matrix diagonal, an independent recount of test accuracy, and trying bad files in the app.

## Prompts that materially shaped the solution

### 1. Planning: which structure and stack (14 Oct)
> "Tell me which structure we shall follow to get the best out of both the models, tell me what all we shall use, and I will select from them." *(with the assignment PDF attached)*

**Output:** inspected the dataset zip, found there is no separate test folder, and proposed using Kaggle `validation/` as the held-out test set and splitting Kaggle `train/` 80/20 into train and val. It then offered options for the baseline (small CNN vs LBP+SVM), the pretrained model (ResNet18 / MobileNetV2 / EfficientNet-B0), the framework and app extras.
**My decision:** PyTorch, small CNN baseline, ResNet18 only.

### 2. Why two models?
> "Why did they say to use 2 models, what do they expect? ... we want more accuracy or confidence rate."

**Output:** the baseline sets a reference point, so the pretrained model's gain can be measured fairly (same split, seed and metrics), and the two can be compared on trade-offs (size, latency). It also pointed out that high confidence is not the same as *calibrated* confidence, and suggested temperature scaling fitted on val.

### 3. Prioritising scope (my review of the plan)
reviewed the proposed stack and decided:
1. ResNet18 as the only pretrained model. MobileNet only if time is left.
2. Grad-CAM as a bonus, not core.
3. Simple DVC first (track data + pipeline), not over-engineered.
4. Temperature scaling only after basic predictions work, and validated on held-out data.
5. pytest first. GitHub Actions only if time allows.

### 4. Implementation
> "Build the core repo: data split, both models, training with MLflow, evaluation, inference module, Streamlit app with input validation, tests, DVC pipeline, README."

### 5. Debugging the setup on my Mac (14 Oct)
> *(screenshots of Terminal errors)* "now shall I do step 2?" 

Three real problems came up when I first ran the code:
1. **zsh treated `#` comments as arguments** (`zsh: number expected`). Fix: `setopt interactivecomments`, or paste commands without comments.
2. **ResNet18 weights failed to download** (SSL certificate error, common with python.org Python on macOS). The suggested `Install Certificates.command` did not exist on my Mac, so the fix was `pip install certifi` and `export SSL_CERT_FILE=$(python -m certifi)`. After one download the weights are cached in `~/.cache/torch/`.
3. **A Pillow deprecation warning** in the 16-bit image test (the `mode=` argument is being removed). The test was updated to the new API.

**My check:** re-ran `pytest` (all passing) and the `--quick` training run (2 epochs, 21.5 s on Apple MPS, 99.65% val accuracy) before running the full `dvc repro`.

### 6. Understanding the pipeline
> "So training, testing and validation is all in step 1?" / "But this is all one model, right? Then we need to compare it with the pretrained one."

**Output:** step 1 is only a smoke test; `dvc repro` runs every stage once per model (`foreach: [baseline, resnet18]` in `dvc.yaml`). Validation happens inside training (choosing the best epoch, early stopping); the test set is used once, at the end, by `src/evaluate.py`.

### 7. Checking where the accuracy comes from
> "But wait, how do I know the model's accuracy is so and so?"

**Output:** accuracy = correct / 360 on the held-out test set. Ways to verify it: the saved `metrics.json`, adding up the confusion-matrix diagonal by hand (358 for the baseline, 360 for ResNet18), the MLflow UI, and a short independent script that loops over the test folders and asks the model directly, without using the evaluation code.

### 8. Ownership and scope
> "I clearly said let's both do it, but it seems like you finished the model. Don't do anything apart from what they asked, even the dataset, don't generate anything new."

**Output:** an exact account of what the AI did (wrote the code and the first draft of the docs; unzipped the original Kaggle dataset without changing any image; created a few deliberately broken files outside the project only to test the app's error messages; copied 6 original test images into `samples/` for the demo) and what I did (installed everything, ran the tests and the full `dvc repro` that trained and evaluated both models, set up Git and DVC, tested the app).
**My decision:** no further changes to my project without my say-so; nothing outside the brief (no chat feature in the app, no extra models); I go through the code module by module so I can explain and change it myself.

### 9. Review against the brief 
> "Check the work I did and create a checklist of what all I need to do according to the assignment, what is already done and what needs to be done, don't miss anything."

**Output:** a requirement-by-requirement check of the repo against the PDF. It found gaps: `requirements-lock.txt` was referenced but missing, the README said Python 3.11 while my environment is 3.13, the DVC remote is a local folder so `dvc pull` cannot work for reviewers, MLflow results were not visible in the repo (`mlruns/` is gitignored), and this prompt log still had a placeholder.
**My decision:** fix all of these before submitting; add MLflow screenshots; explain the local DVC remote in the README instead of setting up a cloud remote (not worth the time for a 1,800-image dataset).

### 10. Packaging
> "Let's finish everything first, and then learn."

**Output:** generated `requirements-lock.txt` from my virtual environment, updated the README (Python version, DVC remote note, pipeline graph, screenshot links), and built the submission ZIP without `.venv`, `.git`, caches, MLflow runs or the raw dataset.

### 11. Experiment-tracking evidence, GitHub and deployment (14 Oct)
> "Where [do I find the runs]?" / "All are not fitting" (screenshots of the MLflow UI) and step-by-step help pushing to GitHub and deploying on Streamlit Community Cloud.

**Output:** how to switch MLflow 3 to the "Model training" view, which metric columns to show, and how to read the loss curves (the drop at step 5 is where the ResNet18 backbone is unfrozen). Also how to create the GitHub repo and deploy `app.py` on Streamlit Cloud.
**My check:** compared the MLflow values with `reports/comparison.md` (test accuracy 1.0 vs 0.9944, best val accuracy 1.0 vs 0.9931); after deploying, tested the live app with uploaded images and invalid files.


---

### 12. Understanding the training loop (15 Oct)
> "Walk me through `train.py`: what happens in one epoch?"

**Output:** one epoch = 36 shuffled, augmented batches of 32 (forward pass → cross-entropy loss → backward → AdamW step), then a validation pass with no weight updates, a learning-rate scheduler step, MLflow logging, and the early-stopping check on val loss. The best epoch (not the last) is restored and saved. ResNet18 runs this loop twice (frozen head, then full fine-tuning).

### 13. Data leakage
> "What is data leakage, and how does my project prevent it?"

**Output:** five safeguards in my code: a separate test set read only by `evaluate.py`; every decision (early stopping, calibration, model choice) made on val; MD5 check that drops any training image identical to a test image; identical images kept in the same split; and assertions that crash if any image appears in two splits. Also: normalisation uses fixed ImageNet statistics, not statistics computed from my data. Known gap: near-duplicates are not caught by MD5.

### 14. Reading the confusion matrix, and accuracy vs precision vs confidence
> "What does my confusion matrix show? The app often shows ~100% confidence. Is the model more accurate or more precise, and how is that different from confidence?"

**Output:** rows are true classes and columns are predicted classes; the diagonal holds correct predictions (baseline: 358 of 360). Precision, recall and accuracy are read from the same matrix: the baseline's inclusion→scratches error lowers *scratches precision* (60/61) and *inclusion recall* (59/60). For ResNet18 all three are 1.0, so it is not "more accurate than precise". Confidence is a different thing: the softmax probability for one image, which says how sure the model is, not whether it is right (the baseline was 96% confident on a wrong prediction).
**My check:** added up the diagonals myself (358 and 360) and matched them with `metrics.json`.

## How I validated the AI-generated output
- **Data split:** ran `prepare_data` on the real dataset and checked the counts (1,152 / 288 / 360, balanced). Confirmed the duplicate pair `patches_101`/`patches_105` by MD5 and that it stays in one split.
- **Tests:** ran `pytest`. They cover validation, the split, metrics/calibration, preprocessing, model shapes and the Predictor.
- **Training:** watched the train/val curves for over-fitting; checked that the best epoch was chosen on val loss, not test.
- **Evaluation:** cross-checked the accuracy in `metrics.json` against the confusion-matrix diagonal by hand; looked at every misclassified image.
- **App:** tried valid images from every class, plus a colour photo, a PDF renamed to `.jpg`, an empty file and a GIF, and confirmed each message.
- **Understanding:** before submitting I went through each module with the AI, asked it to explain the parts I did not understand, and checked those explanations against the code and the outputs.


