# Windows Contribution Setup Guide

## Prerequisites

### 1. Install uv
```cmd
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Rust (required for building native packages from source)
```cmd
winget install Rustlang.Rustup
```
> Close and reopen your terminal after this step so `rustc` is on PATH.

### 3. Install Visual Studio C++ Build Tools (required only if using Python 3.14+)
```cmd
winget install Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```
> Skip this step if using Python 3.10–3.13 (pre-built wheels are available).

---

## Setup

### 4. Clone your fork and enter the directory
```cmd
git clone https://github.com/<your-username>/dspy.git
cd dspy
```

### 5. Add the upstream remote
```cmd
git remote add upstream https://github.com/stanfordnlp/dspy.git
git remote -v
```

### 6. Install Python 3.10 via uv
```cmd
uv python install 3.10
```

### 7. Sync all dependencies using Python 3.10
```cmd
uv sync --all-extras --python 3.10
```

### 8. Activate the virtual environment
```cmd
.venv\Scripts\activate.bat
```

### 9. Install pre-commit hooks
```cmd
uv run pre-commit install
```

---

## Syncing Your Fork with Upstream

### Sync your local main branch
```cmd
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Rebase your feature branch onto updated main
```cmd
git checkout your-feature-branch
git rebase main
```
