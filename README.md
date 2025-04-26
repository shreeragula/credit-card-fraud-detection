# Credit Card Fraud Detection

Detect fraudulent transactions using machine learning on real-world anonymized credit card data.

---

## Objective

To build and evaluate machine learning models that can accurately identify fraudulent credit card transactions while minimizing false positives.

---

##  Project Structure

| File Name             | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `credit_card.ipynb`  | Jupyter notebook with full code, analysis, and visualizations |
| `model_report.pdf`   | Summary of steps, visualizations, and results                 |
| `README.md`          | Project overview (this file)                                 |
| `.gitignore`         | Files/folders ignored by Git                                  |

---

##  Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions  
- **Fraud cases**: 492 (0.17%)  
- **Features**:
  - `Time`, `Amount` (original features)
  - `V1` to `V28` (PCA-anonymized features)
  - `Class`: 0 = Legitimate, 1 = Fraud

---

##  Steps Followed

### 1. Data Preprocessing

- Verified no missing values
- Dropped `Time` column (not useful)
- Standardized `Amount` using `StandardScaler`

### ⚖ 2. Handling Class Imbalance

- Severe imbalance: ~0.17% fraud cases
- Applied **RandomUnderSampler** to create a balanced 50:50 dataset
- Visualized before and after class distribution

### 3. Train-Test Split

- 80% Training, 20% Testing
- **Stratified split** to maintain class ratios

---

## Models Used

###  Random Forest Classifier
- `n_estimators = 200`
- `random_state = 42`
- Evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC

### XGBoost Classifier
- `n_estimators = 300`
- `max_depth = 6`
- `learning_rate = 0.05`
- `eval_metric = 'logloss'`
- `use_label_encoder=False`
- Showed better recall and F1 Score compared to Random Forest

---

##  Evaluation Metrics

| Metric        | Random Forest | XGBoost |
|---------------|----------------|---------|
| Accuracy      | ✅ High         | ✅ High  |
| Precision     | ✅ Excellent    | ✅ Excellent |
| Recall        | ✅ Moderate     | ✅ Higher  |
| F1 Score      | ✅ Good         | ✅ Better  |
| ROC-AUC Curve | ✅ Strong       | ✅ Very Strong |

- Used **Confusion Matrix**, **ROC Curve**, and **AUC Score** for model evaluation

---

## 🏁 Final Results

The models were evaluated using standard classification metrics.  
Below are the **detailed performance metrics for the XGBoost model**, which showed the best results:

| Metric     | Value      |
|------------|------------|
| **Accuracy**   | 95.95%     |
| **Precision**  | 97.80%     |
| **Recall**     | 89.90%     |
| **F1 Score**   | 93.68%     |

✅ These metrics show that the model:

- Achieves **high precision**, meaning it rarely flags non-fraud as fraud.
- Maintains **strong recall**, successfully identifying most fraudulent transactions.
- Has a **balanced F1 Score**, making it suitable for real-world fraud detection systems.

---

##  Conclusion

- **Random Forest**:
  - Excellent precision but slightly lower recall.
- **XGBoost**:
  - Higher recall and slightly better F1 Score.
  - **Best choice** for scenarios where catching frauds is critical, even if it allows a few more false positives.

---
