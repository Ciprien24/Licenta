import json
with open("runs/cnn_busi_baseline_20260224_165051/split.json") as f:
    s = json.load(f)
train = set(s["train"])
val = set(s["val"])
print("train:",len(train))
print("val:",len(val))
print("overlap:",len(train & val))
