# ML & Deep Learning — Course Project

**School of Artificial Intelligence and Data Science | 2024–2025**

Total Points: 60 | Tasks: 6 | Framework: PyTorch + scikit-learn

---

## Project Structure

```
ProjectGroup_[ID]/
├── notebooks/
│   ├── task1_classification_algorithms.ipynb   # Task 1 — Theory & Deep Dive
│   ├── task2_implementation.ipynb              # Task 2 — Python Pipeline
│   ├── task3_dimensionality_reduction.ipynb    # Task 3 — PCA, t-SNE, UMAP, LDA
│   ├── task4_deep_learning.ipynb               # Task 4 — MLP, Optimizers
│   ├── task5_cnn.ipynb                         # Task 5 — CNN, Transfer Learning
│   └── task6_overview_integration.ipynb        # Task 6 — Survey & Conclusions
├── src/
│   ├── preprocessing.py                        # Data loading & preprocessing
│   ├── models.py                               # ML & DL model definitions
│   ├── evaluation.py                           # Metrics & visualization utilities
│   └── train_cnn.py                            # CNN training script
├── results/                                    # Saved models, figures, logs
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Install Python 3.10+
Download from https://python.org if not already installed.

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU support (optional):** If you have an NVIDIA GPU, install the CUDA-enabled PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## How to Run

### Option A — Jupyter in VS Code (recommended)
1. Open VS Code in the project root folder
2. Open any `.ipynb` file from the `notebooks/` folder
3. Select the Python interpreter (top right of notebook)
4. Run all cells: **Kernel → Restart & Run All**

### Option B — Jupyter Lab in browser
```bash
jupyter lab
```
Then open notebooks from the file browser.

### Recommended run order
```
task1 → task2 → task3 → task4 → task5 → task6
```
Tasks 3–6 depend on data/models produced in Task 2.

---

## Datasets

| Task | Dataset | Source | Auto-download? |
|------|---------|--------|----------------|
| Task 2, 3, 4 | Breast Cancer (UCI) | `sklearn.datasets` |  Yes |
| Task 2 | Wine Quality | UCI Repository |  Yes (or synthetic fallback) |
| Task 5 | MNIST | `torchvision.datasets` |  Yes (~11MB) |

All datasets download automatically on first run. No manual setup needed.

---

## Results

After running all notebooks, the `results/` folder will contain:

| File | Description |
|------|-------------|
| `task1_decision_boundaries.png` | Decision boundary comparison |
| `task1_overfit_underfit.png` | Overfitting/underfitting analysis |
| `task2_*_confusion.png` | Confusion matrices per dataset |
| `task2_*_roc.png` | ROC curves per dataset |
| `task2_model_comparison.png` | All model metrics bar chart |
| `task3_pca_variance.png` | PCA explained variance elbow plot |
| `task3_2d_projections.png` | PCA / t-SNE / UMAP / LDA visualizations |
| `task3_pca_comparison.png` | Full vs PCA-reduced accuracy |
| `task4_mlp_relu_history.png` | MLP training curves |
| `task4_optimizer_comparison.png` | SGD vs Adam vs RMSProp |
| `task5_custom_cnn_history.png` | CNN training curves |
| `task5_filters.png` | Learned conv filters |
| `task5_feature_maps.png` | Activation maps |
| `task5_predictions.png` | Correct/incorrect samples |
| `task5_cnn_comparison.png` | Custom CNN vs ResNet-18 |
| `task6_integrated_comparison.png` | Final comparison across all tasks |
| `best_model_A.pkl` | Best classical model (Dataset A) |
| `task4_mlp_relu.pth` | MLP weights |
| `task5_custom_cnn.pth` | Custom CNN weights |
| `task5_resnet18.pth` | Fine-tuned ResNet-18 weights |

---

## Loading Saved Models

```python
# Load MLP
import torch
from src.models import MLP
model = MLP(in_features=30, n_classes=2)
model.load_state_dict(torch.load('results/task4_mlp_relu.pth'))
model.eval()

# Load Custom CNN
from src.models import CustomCNN
cnn = CustomCNN(n_classes=10)
cnn.load_state_dict(torch.load('results/task5_custom_cnn.pth'))
cnn.eval()

# Load best classical model
import pickle
with open('results/best_model_A.pkl', 'rb') as f:
    clf = pickle.load(f)
```

---

## Contribution Statement

| Member | Responsibilities |
|--------|-----------------|
| Yesbol Tynyshtykbay Alimzhan Kuandykov | All tasks: theory, implementation, evaluation, report writing |

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- scikit-learn: https://scikit-learn.org
- PyTorch: https://pytorch.org
