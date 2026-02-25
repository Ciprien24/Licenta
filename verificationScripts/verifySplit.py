import json
from collections import Counter
with open("runs/cnn_busi_baseline_20260225_162715/split.json") as f:
    s = json.load(f)
train = set(s["train"])
val = set(s["val"])
print("train:",len(train))
print("val:",len(val))
print("overlap:",len(train & val))

with open("runs/cnn_busi_baseline_20260225_162715/split.json") as f:
    s = json.load(f)

print("train", len(s["train"]))
print("val", len(s["val"]))
print("test",len(s["test"]))
print("total", len(s["train"])+len(s["val"])+len(s["test"]))


with open("runs/cnn_busi_baseline_20260225_162715/split.json") as f:
    s = json.load(f)
    print(Counter(s["y_train"]))
    print(Counter(s["y_val"]))
    print(Counter(s["y_test"]))
