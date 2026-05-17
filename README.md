# Pattern Recognition & Machine Learning (Graduate Level)

This repository contains the theoretical and practical implementations developed for the **Pattern Recognition** course within the Graduate Program in Computer Science (PPGCC).

The core objective of this curriculum is to consolidate the mathematical and statistical foundations of Machine Learning by building classifiers **From Scratch**. By reconstructing each algorithm using only native functions and matrix manipulation via NumPy (without high-level frameworks like *scikit-learn*), a rigorous understanding is gained regarding parameter estimation, covariance structures, likelihood computation, and numerical stability in high-dimensional spaces.

---

## Technical Methodology

Each classification model follows a rigorous scientific validation pipeline structured as follows:

1. **Robust Data Imputation**: Feature strings and missing tokens (such as `'?'` found in specific clinical datasets) are converted to numerical formats (`errors='coerce'`) and systematically imputed using the mean of the valid column values, avoiding propagation of null types.
2. **Multi-Realization Evaluation**: Performance metrics are evaluated across **25 independent realizations** with randomized splits (80% training / 20% testing) to mitigate partitioning bias and provide realistic confidence intervals (Mean $\pm$ Standard Deviation).
3. **Dimensionality Selection (Fisher-like Criterion)**: For 2D plotting constraints, optimal feature pairs are dynamically chosen based on the ratio of between-class variance (distance between centroids) to within-class variance (cluster compactness).
4. **Boundary Visualization**: Decision surfaces are mapped across a dense 2D grid pixel-by-pixel, extracting exact boundaries where adjacent predicted classes transition. Ellipsoidal probability density contours are overlaid natively to visualize class dispersion.

---

## Repository Structure

```text
.
├── assignments/            # Problem statements and academic specifications
├── courses/                # Theoretical slides and lecture materials
├── implementation/         # Source code (.py) of the algorithms built from scratch
├── latex/                  # LaTeX source code for reports, figures, and academic logs
├── reports/                # Final project documentation compiled in PDF format
└── requirements.txt        # Environment dependencies (NumPy, Pandas, Matplotlib, UCIML)

```

---

## Implemented Classifiers

### 01. Univariate & Bivariate Exploratory Analyzer

* **Path**: `implementation/01-univariado-bivariado.py`
* **Core Concepts**: Initial investigation into data distribution shapes, probability density functions, and simple feature projections.

### 02. Multivariate Gaussian Bayesian Classifier

* **Path**: `implementation/02-multivariada.py`
* **Core Concepts**: Generative modeling of classes via multivariate Gaussian distributions using full covariance matrices ($\Sigma_{d \times d}$), capturing cross-feature correlations.
* **Comparative Baselines**: Minimum Distance to Centroid (DMC) and K-Nearest Neighbors (KNN) are integrated within the same validation pipeline for performance benchmarking.

### 03. Naive Bayes Classifier

* **Path**: `implementation/03-naive-bayes.py`
* **Core Concepts**: Enforces the conditional independence assumption. The model optimizes computational complexity by reducing full covariance matrices to diagonal structures, evaluating joint likelihood as a product of independent 1D Gaussian densities.

### 04. Linear & Quadratic Discriminant Analysis (LDA / QDA)

* **Path**: `implementation/04-discriminantes-linear-quadratico.py`
* **Core Concepts**: Analytical decision boundaries derived from assumptions regarding class covariance homogeneity. LDA shares a pooled covariance matrix across all classes, yielding linear hyperplanes, whereas QDA estimates individual covariance structures, generating quadratic quadratic boundaries.

---

## Benchmarked Datasets

The implementations are verified against real-world classification tasks sourced from the UCI Machine Learning Repository:

1. **Iris Flower**: A classic, linearly separable baseline with low dimensionality (4 features, 3 classes).
2. **Vertebral Column**: A biomedical task focusing on the classification of spinal pathologies (6 features, 3 classes).
3. **Artificial I**: A controlled synthetic dataset drawn from predefined normal distributions, used to validate the accuracy of decision boundaries and probability ellipses.
4. **Breast Cancer**: A diagnostic task presenting challenges in handling mixed discrete features and clinical target balances.
5. **Dermatology**: A highly dimensional clinical dataset featuring **34 attributes** and small minority classes (e.g., Phase 6 Pityriasis Rubra Pilaris). This data profile tests model resilience against matrix singularity and zero-variance columns, handled through adaptive ridge-type diagonal regularization ($\Sigma + \epsilon I$).

---

## Execution Guide

1. Initialize and activate your virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate

```


2. Install the bare minimum framework dependencies:
```bash
pip install -r requirements.txt

```


3. Execute any specific pipeline from the root directory:
```bash
python implementation/02-multivariada.py

```
---

**Author:** Raquel Maciel Coelho de Sousa

*Graduate Program in Computer Science (PPGCC)*