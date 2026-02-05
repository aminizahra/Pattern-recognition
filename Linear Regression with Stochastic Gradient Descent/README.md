# Linear Regression with Stochastic Gradient Descent (SGD)

A robust implementation of **Linear Regression** optimized using **Stochastic Gradient Descent (SGD)**. This project demonstrates the fundamental concepts of Pattern Recognition and Machine Learning, focusing on iterative optimization to minimize the cost function efficiently on datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-Stochastic%20Gradient%20Descent-orange.svg)
![Status](https://img.shields.io/badge/Status-Maintained-green.svg)

## 📌 Overview

In classical Linear Regression, the goal is to fit a linear model to a set of observed data. While **Batch Gradient Descent** computes the gradient using the entire dataset for every step, this approach becomes computationally expensive with large datasets.

**Stochastic Gradient Descent (SGD)** addresses this by updating the model parameters for **each training example** one by one. This results in faster convergence for large datasets and adds a stochastic element that can help the model escape local minima.

### Key Features
* **Pure Python/NumPy implementation** of the SGD algorithm.
* **Visualization** of the regression line evolution.
* **Mathematical breakdown** of the optimization process.
* **Performance metrics** (Cost/Loss tracking).

---

## 🧮 Mathematical Formulation

### 1. Hypothesis Function
The linear relationship is modeled as:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Where:
* $x$ is the input feature.
* $\theta_0$ (bias) and $\theta_1$ (weight) are the parameters to be learned.

### 2. Cost Function (Mean Squared Error)
To measure the accuracy of our hypothesis, we use the Mean Squared Error (MSE) cost function:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### 3. Stochastic Gradient Descent Update Rule
Unlike Batch Gradient Descent which sums over all $m$ examples, SGD updates the parameters $\theta_j$ for **each** training example $(x^{(i)}, y^{(i)})$ individually:

$$\theta_j := \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

Where:
* $\alpha$ represents the **Learning Rate**.
* The update is performed iteratively for a specified number of **Epochs**.

---

## 📊 Results

The plot below visualizes the dataset (blue dots) and the final regression line (red line) learned by the SGD algorithm.

![SGD Result Plot](https://raw.githubusercontent.com/aminizahra/Pattern-recognition/refs/heads/main/Linear%20Regression%20with%20Stochastic%20Gradient%20Descent/Final%20Result%20SGD.png)

*Figure 1: The fitted line demonstrates the convergence of the stochastic gradient descent algorithm on the input data.*

---

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python installed along with the required libraries:

```bash
pip install numpy matplotlib pandas
```
### Running the Project
Clone the repository:

```bash
git clone [https://github.com/aminizahra/Pattern-recognition.git](https://github.com/aminizahra/Pattern-recognition.git)
```

### Navigate to the project directory:

```bash
cd "Pattern-recognition/Linear Regression with Stochastic Gradient Descent"
```

### Run the script:
```bash
python main.py
# Note: Replace 'main.py' with the actual name of your script if different.
```

### 👤 Author: Zahra Amini

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="30" alt="GitHub Logo"> [GitHub: @aminizahra](https://github.com/aminizahra)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/globe.svg" width="30" alt="Portfolio Logo"> [Portfolio](https://aminizahra.github.io/)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/linkedin.svg" width="30" alt="LinkedIn Logo"> [LinkedIn](https://www.linkedin.com/in/zahraamini-ai/)
