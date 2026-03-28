Purpose
This file gives concise, actionable guidance for AI coding agents working on this X-ray classification repo.

Quick context
- The repo uses a bundled virtual environment at `xray-cv/` and a pinned `requirements.txt` with `torch`, `torchvision`, `albumentations`, and common science stack.
- Dataset is organized as image-class folders under `train/` and `test/` (e.g. `train/BACTERIAL_PNEUMONIA/`, `train/NORMAL/`, `train/VIRAL_PNEUMONIA/`) — ImageFolder-compatible layout.
- Top-level `src/` contains module placeholders: `src/eda.py`, `src/dataset.py`, `src/model.py`, `src/train.py` (currently empty in this snapshot). `notebooks/` holds exploratory work.
- Model artifacts are expected under `models/`.

Immediate priorities for changes
- Preserve the ImageFolder-style layout when writing data loaders (use `torchvision.datasets.ImageFolder` or equivalent).
- Use the included virtualenv `xray-cv/bin/python` for reproducible runtime commands in examples and CI.

Useful commands (run from repository root)
- Activate the bundled venv: `source xray-cv/bin/activate` (or reference `xray-cv/bin/python` directly).
- Install dependencies (use the venv python): `xray-cv/bin/python -m pip install -r requirements.txt`.
- Launch a quick interactive check: `xray-cv/bin/python -c "import torch, torchvision; print(torch.__version__)"`.

Code & testing conventions
- Expect training/experiment code in `src/` or `notebooks/`. If you add a CLI, keep a minimal entrypoint in `src/train.py` that imports `src.dataset` and `src.model` and writes checkpoints to `models/`.
- Prefer saving PyTorch checkpoints with `torch.save({'state_dict': model.state_dict(), ...}, path)` into `models/`.
- No automated tests are present—include focused unit tests for data transforms and model I/O if adding CI.

Patterns to follow when editing
- Data transforms: follow `albumentations`/`torchvision.transforms` style already listed in `requirements.txt`.
- Use deterministic seeds where feasible for reproducibility (set `torch.manual_seed` and numpy seed in training scripts).
- Keep heavy experiments in `notebooks/` and production/trainable scripts in `src/`.

Integration & environment notes
- The repo bundles a venv under `xray-cv/`; prefer using that exact interpreter for reproducing results and examples.
- There are no external services (DBs/APIs) discovered — model training is local and filesystem-driven.

References (key paths)
- `requirements.txt` — pinned deps and key libs.
- `train/` and `test/` — dataset folders with class subdirectories.
- `src/` — code modules (placeholders currently).
- `models/` — target for saved checkpoints.

If anything in this file looks incorrect or incomplete, tell me which files or workflows you'd like me to inspect next and I will update this guidance.
