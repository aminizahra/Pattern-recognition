# Pattern Recognition & Machine Learning Implementations

A comprehensive collection of fundamental Machine Learning algorithms implemented from scratch. This repository covers a wide spectrum of techniques, from classical linear regression to generative probabilistic models and unsupervised clustering.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Implementations-purple.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

## 📁 Repository Structure

The projects are organized by algorithm type and complexity. Each directory contains a detailed implementation, mathematical background, and performance visualizations.

### 1. Regression Analysis
* **[Linear Regression (Closed-Form)](./Linear%20Regression_Closed%20Form/)**: Solving linear models analytically using the Normal Equation.
* **[Linear Regression (SGD)](./Linear%20Regression%20with%20Stochastic%20Gradient%20Descent/)**: Efficient iterative optimization using Stochastic Gradient Descent for large-scale data.

### 2. Classification Models
* **[Binary Logistic Regression](./Binary%20Classification%20Logistic%20regression/)**: Probabilistic binary classification using the Sigmoid activation and Log-Loss optimization.
* **[Multiclass Softmax Regression](./Multiclass%20Classification_Softmax/)**: Extending logistic regression to multiple classes using Softmax, OvA, and OvO strategies.
* **[Bayesian GLDA](./Bayesian%20Classification/)**: Gaussian Linear Discriminant Analysis for generative classification with shared covariance.
* **[Quadratic Discriminant Analysis (QDA)](./Quadratic%20Multiclass%20Classification/)**: Non-linear multiclass classification with class-specific covariance matrices.
* **[Naïve Bayes](./Naïve%20Bayes%20Classification/)**: Sentiment analysis on text data (Yelp, IMDB, Amazon) using probabilistic word frequencies and Laplace smoothing.

### 3. Unsupervised Learning
* **[K-Means Clustering](./Kmeans%20on%20Image%20Compression/)**: Application of K-Means for image compression and color quantization (Vector Quantization).

---

## 🧮 Core Concepts Explored


This repository serves as a practical guide to the mathematical foundations of Pattern Recognition:
* **Optimization:** Gradient Descent vs. Analytical Closed-Form solutions.
* **Generative vs. Discriminative:** Modeling class distributions ($P(x|y)$) vs. direct boundary learning ($P(y|x)$).
* **Linear vs. Non-Linear:** Understanding when to use linear separators (LDA/Logistic) versus quadratic surfaces (QDA).
* **Natural Language Processing:** Tokenization and Bag-of-Words modeling for sentiment prediction.

---

## 🛠️ Requirements & Setup

### Prerequisites
Ensure you have Python 3.x installed. The following libraries are used across various projects:
* `numpy` & `scipy`: Matrix operations and numerical computing.
* `pandas`: Data manipulation and analysis.
* `matplotlib` & `seaborn`: Data visualization and 3D plotting.
* `scikit-learn`: Used primarily for data splitting and evaluation metrics.
* `nltk`: Natural language processing tools for Naïve Bayes.

### Installation
```bash
git clone https://github.com/aminizahra/Pattern-recognition.git
cd Pattern-recognition
pip install -r requirements.txt # Or install the libraries listed above
```
---

### 👤 Author: Zahra Amini

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="30" alt="GitHub Logo"> [GitHub](https://github.com/aminizahra)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/globe.svg" width="30" alt="Portfolio Logo"> [Portfolio](https://aminizahra.github.io/)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/linkedin.svg" width="30" alt="LinkedIn Logo"> [LinkedIn](https://www.linkedin.com/in/zahraamini-ai/)

---
Note: This repository was created for educational purposes to demonstrate the "from-scratch" implementation of machine learning algorithms.
