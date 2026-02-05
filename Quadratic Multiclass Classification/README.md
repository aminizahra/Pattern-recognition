# Quadratic Multiclass Classification (QDA)

This project implements **Quadratic Discriminant Analysis (QDA)** for multiclass classification. Unlike Linear Discriminant Analysis (LDA), QDA allows for non-linear decision boundaries by assuming that each class has its own unique covariance matrix, making it highly effective for complex, overlapping datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-QDA-orange.svg)
![Type](https://img.shields.io/badge/Task-Multiclass%20Classification-green.svg)

## 📌 Overview

Quadratic Discriminant Analysis (QDA) is a generative model that assumes the input data $X$ follows a Gaussian distribution for each class. It is particularly powerful when the decision boundary between classes is not a straight line (non-linear).

### Key Features
* **Generative Modeling:** Estimates the probability distribution of each class.
* **Flexible Boundaries:** Supports quadratic decision surfaces.
* **3D Visualization:** Includes Probability Density Function (PDF) surface plots.
* **Performance Analysis:** Comprehensive evaluation using accuracy scores and classification reports.

---

## 🧮 Mathematical Formulation

### 1. Multivariate Gaussian Distribution
For each class $k$, the density function is defined as:

$$f_k(x) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

Where:
* $\mu_k$ is the mean vector of class $k$.
* $\Sigma_k$ is the **class-specific** covariance matrix.

### 2. Quadratic Discriminant Function
The decision rule is derived by maximizing the posterior probability $P(G=k | X=x)$. The quadratic score function for class $k$ is:

$$\delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k$$

Where $\pi_k$ is the prior probability of class $k$. The boundary between two classes exists where $\delta_k(x) = \delta_l(x)$.

---

## 📊 Results & Visualizations

Below are the visual evaluations and performance reports generated during the training and testing phases across multiple datasets.

### Decision Boundaries & Scatter Plots
These plots visualize how the QDA model separates different classes in a 2D space.

| Feature Map / Boundary | Training & Testing Plots |
| :---: | :---: |
| ![Plot 1](R%20(1).jpg) | ![Plot 2](R%20(2).jpg) |
| ![Plot 3](R%20(3).jpg) | ![Plot 4](R%20(4).jpg) |
| ![Plot 5](R%20(5).jpg) | ![Plot 6](R%20(6).jpg) |

### 3.D PDF and Contours
Visualizing the Gaussian distributions for the modeled classes.

| 3D Probability Density | Contour Plots |
| :---: | :---: |
| ![Plot 7](R%20(7).jpg) | ![Plot 8](R%20(8).jpg) |

### Classification Reports
Detailed performance metrics including Precision, Recall, and F1-score for each class.

| Dataset 1 Report | Dataset 2 Report |
| :---: | :---: |
| ![Report 9](R%20(9).jpg) | ![Report 10](R%20(10).jpg) |

---

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have the following packages installed:
```bash
pip install numpy matplotlib pandas scipy scikit-learn
```
### Running the Notebook
1. Clone the repository.

2. Open the notebook in your preferred environment:

```bash
jupyter notebook QDA.ipynb
```
3. Run all cells to see the training process and generated plots.

---

### 👤 Author: Zahra Amini

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="30" alt="GitHub Logo"> [GitHub](https://github.com/aminizahra)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/globe.svg" width="30" alt="Portfolio Logo"> [Portfolio](https://aminizahra.github.io/)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/linkedin.svg" width="30" alt="LinkedIn Logo"> [LinkedIn](https://www.linkedin.com/in/zahraamini-ai/)
