# MACHINE LEARNING – DETAILED SYLLABUS (BEGINNER → ADVANCED)

---

## 1. Introduction to Machine Learning

- What machine learning really is (learning patterns from data, not rule-based systems)
- Traditional programming vs machine learning workflow
- Real-world examples: spam detection, recommendation systems, fraud detection
- Types of ML problems based on output nature
- ML pipeline overview (problem → model → evaluation → improvement)
- Where ML fits in AI, Data Science, and Software Engineering

---

## 2. Types of Learning: Supervised, Unsupervised, Semi-Supervised

- Supervised learning concept with labeled data
- Regression vs classification intuition with examples
- Unsupervised learning idea using hidden structure discovery
- Clustering vs dimensionality reduction use cases
- Semi-supervised learning motivation in real-world data scarcity
- Mapping problem statements to learning types

---

## 3. Regression Fundamentals

- What regression means mathematically and practically
- Continuous target prediction intuition
- Line fitting as an optimization problem
- Error as distance between prediction and truth
- Use cases: price prediction, demand forecasting, risk estimation

---

## 4. Simple Linear Regression

- Linear relationship between input and output
- Equation form:  
$y = wx + b$  

- Meaning of weight (slope) and bias (intercept)
- Visual intuition using straight-line fitting
- Mean Squared Error (MSE) as loss function
- Manual calculation intuition vs automated optimization
- Implementing linear regression using `sklearn`
- Understanding coefficients after training

---

## 5. Multiple Linear Regression

- Extending regression to multiple features
- Equation form:  
$y = w_1x_1 + w_2x_2 + ... + b$
- Feature contribution and coefficient interpretation
- Multicollinearity intuition
- Why adding more features can hurt performance
- Model fitting using matrix operations
- Practical implementation using `sklearn`
- Reading model summary and coefficients

---

## 6. Cost Function and Optimization

- Why models need a cost (loss) function
- Squared error intuition and penalty mechanism
- Convex vs non-convex loss landscapes
- Global minimum vs local minimum
- Relationship between loss minimization and learning

---

## 7. Gradient Descent

- Optimization as “walking downhill”
- Partial derivatives intuition (slope in multidimensional space)
- Gradient vector meaning
- Learning rate and its impact
- Convergence behavior
- Batch Gradient Descent idea
- Visual explanation of parameter updates
- Implementing gradient descent from scratch (basic code logic)

---

## 8. Polynomial Regression

- Why linear models fail on curved data
- Feature transformation intuition
- Polynomial terms without changing linearity in parameters
- Overfitting risk with higher-degree polynomials
- Choosing degree using validation
- Implementation using `PolynomialFeatures`

---

## 9. Overfitting and Underfitting

- Bias vs variance intuition
- Model complexity trade-off
- Training error vs test error behavior
- Visual learning curves explanation
- Detecting overfitting without touching data preprocessing
- Practical signals from metrics and plots

---

## 10. Model Evaluation (Regression)

- Why accuracy is not suitable for regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score intuition
- Choosing metrics based on business context
- Evaluating models using `sklearn`

---

## 11. Regularization (L1 and L2)

- Why models overfit mathematically
- Penalizing large coefficients intuition
- L2 Regularization (Ridge): shrinking weights smoothly
- L1 Regularization (Lasso): feature elimination behavior
- Effect on cost function:  
$Loss + \lambda \sum w^2 \quad (L2)$
- Choosing regularization strength
- Practical implementation with Ridge and Lasso

---

## 12. Bias–Variance Trade-off

- Mathematical intuition behind bias and variance
- High bias vs high variance models
- Relationship with model complexity
- Regularization as bias–variance control
- Diagnosing bias/variance from errors

---

## 13. Introduction to Classification

- Classification as decision boundary learning
- Discrete output prediction
- Probability-based thinking
- Linear vs non-linear decision boundaries
- Binary vs multiclass classification

---

## 14. Logistic Regression (Binary Classification)

- Why linear regression fails for classification
- Sigmoid function intuition:  
$\sigma(z) = \frac{1}{1 + e^{-z}}$
- Probability interpretation of outputs
- Decision threshold concept
- Log Loss (Binary Cross Entropy) intuition
- Gradient descent in logistic regression
- Implementing logistic regression in `sklearn`

---

## 15. Classification Evaluation Metrics

- Confusion Matrix and its components
- Accuracy and its limitations
- Precision, Recall intuition
- F1 Score trade-offs
- Business-driven metric selection
- Metric computation using `sklearn`

---

## 16. Multiclass Logistic Regression

- One-vs-Rest strategy
- Softmax function intuition
- Log loss for multiclass problems
- Handling multiple classes mathematically
- Implementation using `sklearn`

---

## 17. Naive Bayes

- Probabilistic thinking behind classification
- Bayes Theorem intuition:  
$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- Conditional independence assumption
- Gaussian vs Multinomial Naive Bayes
- When Naive Bayes works surprisingly well
- Text classification use cases
- Implementation using `sklearn`

---

## 18. Support Vector Machines (SVM)

- Maximum margin intuition
- Support vectors meaning
- Hard margin vs soft margin
- Kernel trick intuition (non-linear boundaries)
- Hyperplane geometry explanation
- Regularization parameter `C`
- Practical usage and limitations

---

## 19. Decision Trees

- Tree-based decision making intuition
- Recursive splitting logic
- Gini Index vs Entropy
- Information Gain concept
- Overfitting in deep trees
- Interpreting tree structure
- Implementation using `sklearn`

---

## 20. Ensemble Learning Fundamentals

- Why single models fail
- Wisdom of multiple models
- Bias and variance reduction via ensembles
- Voting vs averaging intuition

---

## 21. Bagging and Random Forest

- Bootstrap sampling idea
- Independent model training
- Random Forest intuition
- Feature randomness advantage
- Feature importance interpretation
- Practical Random Forest implementation

---

## 22. Boosting Techniques

- Sequential error correction intuition
- AdaBoost concept
- Gradient Boosting idea
- XGBoost intuition (speed + regularization)
- Bias reduction through boosting
- Practical considerations and tuning

---

## 23. Model Evaluation for Classification

- ROC Curve intuition
- AUC interpretation
- Threshold tuning
- Cost-sensitive decision making
- Using ROC in `sklearn`

---

## 24. Cross Validation Techniques

- Why train-test split is insufficient
- K-Fold Cross Validation
- Stratified K-Fold intuition
- Bias-variance impact of CV
- Practical implementation

---

## 25. Hyperparameter Tuning

- Parameters vs hyperparameters
- Grid Search vs Random Search
- Cross-validation integration
- Overfitting risks during tuning
- Practical tuning workflow

---

## 26. K-Means Clustering

- Unsupervised clustering intuition
- Distance-based grouping
- Centroid update mechanism
- Objective function intuition
- Choosing K using Elbow method
- Limitations of K-Means
- Practical implementation

---

## 27. Variance Inflation Factor (VIF)

- Multicollinearity detection intuition
- Mathematical meaning of VIF
- Impact on regression coefficients
- When to worry about VIF
- Practical interpretation

---

## 28. End-to-End ML Thinking

- Framing ML problems correctly
- Translating business goals to metrics
- Model selection reasoning
- Error analysis mindset
- Iterative improvement philosophy
- Real-world ML pitfalls and best practices

---