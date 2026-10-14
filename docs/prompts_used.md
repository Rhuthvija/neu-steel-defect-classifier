# AI tools and prompt log

**Tool used:** Claude (Anthropic), in the Claude app, linked to my laptop's Downloads folder so it could read the assignment PDF and the dataset zip.

**How I used it:** to plan the approach, draft the code and documentation, and explain design choices. I made the final decisions (listed below), ran all training and evaluation myself on my Mac, and reviewed every file before committing.

---

## Prompts that materially shaped the solution

### 1. Planning: which structure and stack (14 Oct)
> "Tell me which structure we shall follow to get the best out of both the models, tell me what all we shall use, and I will select from them." *(with the assignment PDF attached)*

**Output:** inspected the dataset zip, found there is no separate test folder, and proposed using Kaggle `validation/` as the held-out test set and splitting Kaggle `train/` 80/20 into train and val. It then offered options for the baseline (small CNN vs LBP+SVM), the pretrained model (ResNet18 / MobileNetV2 / EfficientNet-B0), the framework and app extras.
**My decision:** PyTorch, small CNN baseline, ResNet18 only.

### 2. Why two models?
> "Why did they say to use 2 models, what does he expect? ... we want more accuracy or confidence rate."

**Output:** the baseline sets a reference point, so the pretrained model's gain can be measured fairly (same split, seed and metrics), and the two can be compared on trade-offs (size, latency). It also pointed out that high confidence is not the same as *calibrated* confidence, and suggested temperature scaling fitted on val.

### 3. Prioritising scope (my own review of the plan)
I reviewed the proposed stack and decided:
1. ResNet18 as the only pretrained model. MobileNet only if time is left.
2. Grad-CAM as a bonus, not core.
3. Simple DVC first (track data + pipeline), not over-engineered.
4. Temperature scaling only after basic predictions work, and validated on held-out data.
5. pytest first. GitHub Actions only if time allows.

### 4. Implementation
> "Build the core repo: data split, both models, training with MLflow, evaluation, inference module, Streamlit app with input validation, tests, DVC pipeline, README."

### 5. Review against the brief (14 Oct, second Claude account)
> "Check the work I did and create a checklist of what all I need to do according to the assignment, what is already done and what needs to be done, don't miss anything."

**Output:** a requirement-by-requirement check of the repo against the PDF. It found gaps: `requirements-lock.txt` was referenced but missing, the README said Python 3.11 while my environment is 3.13, the DVC remote is a local folder so `dvc pull` cannot work for reviewers, MLflow results were not visible in the repo (`mlruns/` is gitignored), and this prompt log still had a placeholder.
**My decision:** fix all of these before submitting; add MLflow screenshots; explain the local DVC remote in the README instead of setting up a cloud remote (not worth the time for a 1,800-image dataset).

### 6. Packaging
> "Let's finish everything first, and then learn."

**Output:** generated `requirements-lock.txt` from my virtual environment, updated the README (Python version, DVC remote note, pipeline graph, screenshot links), and built the submission ZIP without `.venv`, `.git`, caches, MLflow runs or the raw dataset.

**Note on AI use across accounts:** the project was built in one Claude account and reviewed/packaged in a second one; both are the same tool (Claude).

---

## How I validated the AI-generated output
- **Data split:** ran `prepare_data` on the real dataset and checked the counts (1,152 / 288 / 360, balanced). Confirmed the duplicate pair `patches_101`/`patches_105` by MD5 and that it stays in one split.
- **Tests:** ran `pytest`. They cover validation, the split, metrics/calibration, preprocessing, model shapes and the Predictor.
- **Training:** watched the train/val curves for over-fitting; checked that the best epoch was chosen on val loss, not test.
- **Evaluation:** cross-checked the accuracy in `metrics.json` against the confusion-matrix diagonal by hand; looked at every misclassified image.
- **App:** tried valid images from every class, plus a colour photo, a PDF renamed to `.jpg`, an empty file and a GIF, and confirmed each message.
- **Understanding:** before submitting I went through each module with the AI, asked it to explain the parts I did not understand, and checked those explanations against the code and the outputs.

## External resources
- Dataset: NEU Surface Defect Database (Song & Yan, 2013), Kaggle mirror by bobweng.
- torchvision pretrained ResNet18 weights (ImageNet1K_V1).
- Method references: Grad-CAM (Selvaraju et al., 2017), temperature scaling (Guo et al., 2017).
