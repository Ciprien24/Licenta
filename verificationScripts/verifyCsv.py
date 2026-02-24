import pandas as pd
df = pd.read_csv("runs/cnn_busi_baseline_20260224_165051/preds/val_predictions.csv")
probs_cols = [c for c in df.columns if c.startswith("prob_")]

print(df.head(3))
print("rows: ", len(df))
print("prob sum (first 5:)", df[probs_cols].sum(axis=1).head().tolist())
print("any NaN:", df.isna().any().any())