from mlp import MLP
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import os

FILEPATH = os.path.abspath(__file__)

model_id = "20251005_210459"

df_test = pd.read_parquet(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'testing_data_mlp_{model_id}.parquet'))
df_train = pd.read_parquet(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'training_data_mlp_{model_id}.parquet'))

df_train = df_train.drop(columns=["snored"])
df_train_input_mean = df_train.mean(axis=0)
df_train_input_std = df_train.std(axis=0)

df_test_input = df_test.drop(columns=["snored"])
df_test_input = (df_test_input - df_train_input_mean) / df_train_input_std
df_test_output = df_test[["snored"]]

X_test = df_test_input.to_numpy()
y_test = df_test_output["snored"].to_numpy()[:, np.newaxis]

model = MLP.load(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'mlp_{model_id}.npz'))
y_pred = model.predict(X_test)

accuracy = (y_pred.round() == y_test).mean()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(os.path.join(os.path.dirname(FILEPATH), '..', 'reports', 'figures', f'roc_curve_mlp_{model_id}.png'))
plt.close()

cm = confusion_matrix(y_test, y_pred.round())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Snored", "Snored"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(os.path.dirname(FILEPATH), '..', 'reports', 'figures', f'confusion_matrix_mlp_{model_id}.png'))
plt.close()

print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Test AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Baseline model: Predicts everything as 0
y_baseline = np.zeros_like(y_test)

# Metrics for the baseline model
baseline_accuracy = (y_baseline == y_test).mean()
baseline_auc = roc_auc_score(y_test, y_baseline)
baseline_precision = precision_score(y_test, y_baseline, zero_division=0)
baseline_recall = recall_score(y_test, y_baseline)
baseline_f1 = f1_score(y_test, y_baseline)

print("\nBaseline Always Negative Metrics:")
print(f"Baseline accuracy: {baseline_accuracy*100:.2f}%")
print(f"Baseline AUC: {baseline_auc:.4f}")
print(f"Baseline Precision: {baseline_precision:.4f}")
print(f"Baseline Recall: {baseline_recall:.4f}")
print(f"Baseline F1 Score: {baseline_f1:.4f}")

# Random model: Predicts randomly between 0 and 1
rng = np.random.default_rng(seed=42)
y_random = rng.choice([0, 1], size=y_test.shape)
# Metrics for the random model
random_accuracy = (y_random == y_test).mean()
random_auc = roc_auc_score(y_test, y_random)
random_precision = precision_score(y_test, y_random)
random_recall = recall_score(y_test, y_random)
random_f1 = f1_score(y_test, y_random)
print("\nRandom Model Metrics:")
print(f"Random accuracy: {random_accuracy*100:.2f}%")
print(f"Random AUC: {random_auc:.4f}")
print(f"Random Precision: {random_precision:.4f}")
print(f"Random Recall: {random_recall:.4f}")
print(f"Random F1 Score: {random_f1:.4f}")

# always positive model
y_always_positive = np.ones_like(y_test)
always_positive_accuracy = (y_always_positive == y_test).mean()
always_positive_auc = roc_auc_score(y_test, y_always_positive)
always_positive_precision = precision_score(y_test, y_always_positive)
always_positive_recall = recall_score(y_test, y_always_positive)
always_positive_f1 = f1_score(y_test, y_always_positive)
print("\nAlways Positive Model Metrics:")
print(f"Always Positive accuracy: {always_positive_accuracy*100:.2f}%")
print(f"Always Positive AUC: {always_positive_auc:.4f}")
print(f"Always Positive Precision: {always_positive_precision:.4f}")
print(f"Always Positive Recall: {always_positive_recall:.4f}")
print(f"Always Positive F1 Score: {always_positive_f1:.4f}")