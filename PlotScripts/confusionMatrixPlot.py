import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#Loading predictions

df = pd.read_csv("runs/vit_busi_baseline_20260226_164908/preds/test_predictions.csv")

y_true = df["y_true"].values
y_pred = df["y_pred"].values

class_names = ["Normal", "Benign", "Malignant"]
n_classes = 3

#Confusion matrix

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

disp.plot(cmap = "Blues")
plt.title("ViT Confusion Matrix (Test Set)")
plt.savefig("runs/vit_busi_baseline_20260226_164908/plots/ViT_confusion_matrix.png", dpi=300)
plt.show()
