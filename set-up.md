
---

# STEP 0 — CLEAN STATE CHECK (important)

Close:

* Jupyter Notebook / Lab
* VS Code
* all terminals running Python

(Optional but recommended: restart PC)

---

# GOAL SETUP

You will end with:

* ✔ Conda environment (`cuda`)
* ✔ Python 3.10 (stable for PyTorch)
* ✔ PyTorch with CUDA support
* ✔ Jupyter kernel (`cuda-gpt`)
* ✔ Working GPU check

---

#  STEP 1 — CREATE CLEAN ENVIRONMENT

Open **Anaconda Prompt or CMD**:

```bash id="a1k8ld"
conda create -n cuda python=3.10 -y
```

Activate it:

```bash id="b2v9mc"
conda activate cuda
```

---

#  STEP 2 — INSTALL PYTORCH (CUDA VERSION)

Install official PyTorch build:

```bash id="c3x7qa"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This gives you:

* GPU-enabled PyTorch
* CUDA 12.1 compatible build (most stable)

---

#  STEP 3 — INSTALL JUPYTER KERNEL SUPPORT

```bash id="d4p8ld"
pip install ipykernel jupyter
```

---

# STEP 4 — REGISTER THE KERNEL

```bash id="e5v2mc"
python -m ipykernel install --user --name cuda --display-name "cuda-gpt"
```

---

# STEP 5 — TEST (VERY IMPORTANT)

Start Jupyter:

```bash id="f6x9qa"
jupyter notebook
```

Then in a notebook run:

```python id="g7v8ld"
import sys
print(sys.executable)
```

You should see something like:

```
...anaconda3\envs\cuda\python.exe
```

---

# STEP 6 — TEST PYTORCH + GPU

```python id="h8v1mc"
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

---

# WHAT GOOD OUTPUT LOOKS LIKE

✔ No crash
✔ CUDA available = True
✔ GPU name shows (e.g. NVIDIA RTX 3060)

---

# IF SOMETHING FAILS

### If `torch import crashes`

→ reinstall PyTorch step again

---

### If CUDA = False

→ run:

```bash id="i9q2ld"
nvidia-smi
```

If that fails → GPU driver issue, not PyTorch

---

### If wrong Python version shows (3.13 again)

→ stop and tell me immediately (environment mismatch)

---

# SIMPLE RULE GOING FORWARD

For ML + CUDA:

> ✔ ALWAYS use Python 3.10 or 3.11
> ❌ NEVER 3.12+ / 3.13 yet
> ✔ ALWAYS isolate environments (conda only)

---
