# XRay Project

## Planning docs
- Day 1 (P0 scope): `docs/day1_p0_scope.md`

## Quick start (macOS/zsh)

1) Create and activate virtual environment:

```bash
cd /Users/khanalexandr/Desktop/xray_project
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Run dataset split and smoke-check EDA:

```bash
python src/split_data.py
python src/eda.py --data-path data/interim --skip-samples
```

4) Daily workflow:

```bash
cd /Users/khanalexandr/Desktop/xray_project
source .venv/bin/activate
```

Optional check that Python uses the project environment:

```bash
which python
python -V
pip -V
```
