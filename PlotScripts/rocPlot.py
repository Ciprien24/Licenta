import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os



run_dir = "runs/vit_busi_baseline_20260226_164908"
plots_dir = os.path.join(run_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(os.path.join(run_dir, "preds", "test_predictions.csv"))

y_true = df["y_true"].values
classes = [0, 1, 2]
class_names = ["Normal", "Benign", "Malignant"]

y_true_bin = label_binarize(y_true, classes=classes)
y_scores = df[["prob_normal", "prob_benign", "prob_malignant"]].values

fig, ax = plt.subplots(figsize=(7, 6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ViT ROC Curves (Test Set)")
ax.legend()

out_path = os.path.join(plots_dir, "ViT_roc_curves.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved ROC figure to:", out_path)