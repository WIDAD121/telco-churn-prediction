## Cross-validation on the training set (5 folds)

| Model | PR-AUC | ROC-AUC | Precision at 80% recall | MCC at 80% recall | Best parameters |
|---|---|---|---|---|---|
| Logistic Regression **(selected)** | 0.673 ± 0.017 | 0.850 ± 0.011 | 0.519 | 0.480 | C=3.0 |
| Decision Tree | 0.630 ± 0.031 | 0.829 ± 0.013 | 0.500 | 0.458 | max_depth=8, min_samples_leaf=50 |
| SVM (RBF) | 0.645 ± 0.019 | 0.803 ± 0.017 | 0.447 | 0.407 | C=0.3 |
| Random Forest | 0.664 ± 0.022 | 0.846 ± 0.011 | 0.517 | 0.480 | max_depth=10, min_samples_leaf=5 |
| Gradient Boosting | 0.667 ± 0.022 | 0.847 ± 0.012 | 0.524 | 0.485 | learning_rate=0.05, max_iter=400, max_leaf_nodes=4 |

Highest CV PR-AUC: Logistic Regression. Selected with the one-standard-error rule: Logistic Regression.
Decision threshold chosen on out-of-fold predictions for 80% recall: 0.26 (the threshold that maximises MCC would be 0.38).

## Test set (1,409 customers, used once)

| Threshold | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| 0.26 (chosen, recall target) | 0.736 | 0.502 | 0.786 | 0.613 | 0.452 |
| 0.38 (max MCC) | 0.781 | 0.574 | 0.674 | 0.620 | 0.470 |

ROC-AUC 0.847 (95% CI 0.825-0.868), PR-AUC 0.664 (95% CI 0.614-0.713). Brier score 0.135 (always predicting the base rate: 0.195).

### All candidates on the test set (for reference only, not used for selection)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.847 | 0.664 |
| Decision Tree | 0.838 | 0.638 |
| SVM (RBF) | 0.795 | 0.621 |
| Random Forest | 0.846 | 0.656 |
| Gradient Boosting | 0.842 | 0.658 |
