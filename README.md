# Pattern Recognition & Machine Learning
### Graduate Program in Computer Science (PPGCC)

> Classifiers built **from scratch** — no high-level frameworks. Only NumPy, mathematics, and statistical rigor.

---

## Objective

This repository consolidates the mathematical and statistical foundations of machine learning through the **manual implementation of classifiers**, reconstructing each algorithm using only matrix manipulation via NumPy. The focus is on genuine understanding: parameter estimation, covariance structures, likelihood computation, and numerical stability in high-dimensional spaces.

---

## Repository Structure

```
.
├── assignments/        # Problem statements and academic specifications
├── courses/            # Theoretical slides and lecture materials
├── implementation/     # Algorithm source code (.py)
├── latex/              # LaTeX source for reports and figures
├── reports/            # Final documentation in PDF format
└── requirements.txt    # Dependencies (NumPy, Pandas, Matplotlib, UCIML)
```

---

## Validation Methodology

All models follow the same scientific pipeline:

| Step | Description |
|---|---|
| **Imputation** | Missing tokens (`'?'`) converted to numeric and replaced by the column mean |
| **Multi-realization** | 25 independent runs with random splits (80% train / 20% test) |
| **Feature selection** | Optimal pair chosen via Fisher criterion (between-class variance / within-class variance) |
| **Visualization** | Decision boundaries mapped pixel by pixel + ellipsoidal density contours |

---

## Implemented Classifiers

### 01 · Univariate and Bivariate Exploratory Analysis
`implementation/01-univariado-bivariado.py`

Initial investigation of data distributions, probability density functions, and simple feature projections.

---

### 02 · Multivariate Gaussian Bayesian Classifier
`implementation/02-multivariada.py`

Generative modeling via Gaussian distributions with full covariance matrices ($\Sigma_{d \times d}$), capturing cross-feature correlations. Includes **DMC** (Minimum Distance to Centroid) and **KNN** as comparative baselines.

![Multivariate Gaussian](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Multivariate_Gaussian.png/640px-Multivariate_Gaussian.png)
*Multivariate Gaussian distribution with probability contours*

---

### 03 · Naive Bayes
`implementation/03-naive-bayes.py`

Assumes conditional independence between features. Reduces the full covariance matrix to a diagonal structure, computing joint likelihood as a product of independent 1D Gaussian densities.

---

### 04 · LDA and QDA (Linear and Quadratic Discriminant Analysis)
`implementation/04-discriminantes-linear-quadratico.py`

| Model | Covariance | Decision boundary |
|-------|------------|-------------------|
| **LDA** | Shared (pooled) across classes | Linear hyperplanes |
| **QDA** | Individual per class | Quadratic surfaces |

![LDA vs QDA](https://scikit-learn.org/stable/_images/sphx_glr_plot_lda_qda_001.png)
*Visual comparison between LDA (linear) and QDA (quadratic) boundaries*

---

## Datasets

All sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/):

| Dataset | Features | Classes | Main challenge |
|---|---|---|---|
| **Iris** | 4 | 3 | Classic linearly separable baseline |
| **Vertebral Column** | 6 | 3 | Spinal pathology classification |
| **Artificial I** | — | — | Synthetic data for boundary validation |
| **Breast Cancer** | mixed | 2 | Mixed discrete features, class imbalance |
| **Dermatology** | 34 | 6 | High dimensionality, minority classes, matrix singularity risk — handled via ridge regularization ($\Sigma + \varepsilon I$) |

---

## Getting Started

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any implementation
python implementation/02-multivariada.py
```

---

## Author

**Raquel Maciel Coelho de Sousa**  
Graduate Program in Computer Science (PPGCC)