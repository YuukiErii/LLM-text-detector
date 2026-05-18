import json
from collections import Counter
from pathlib import Path

import numpy as np


# 当前文件路径：llm_text_detector/src/data/inspect_teacher_test.py
# parents[0] = data
# parents[1] = src
# parents[2] = llm_text_detector
PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_path = PROJECT_ROOT / "data" / "raw" / "teacher_test.json"

print("Project root:", PROJECT_ROOT)
print("Input path:", input_path)

if not input_path.exists():
    raise FileNotFoundError(
        f"Cannot find teacher test file at: {input_path}\n"
        f"Please put the JSON file under data/raw/ and rename it to teacher_test.json"
    )

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

word_lengths = [len(t.split()) for t in texts]
char_lengths = [len(t) for t in texts]

print("=" * 60)
print("Teacher Test Set Inspection")
print("=" * 60)

print("Total samples:", len(data))
print("Label distribution:", Counter(labels))

print("\nWord length:")
print("Mean:", np.mean(word_lengths))
print("Median:", np.median(word_lengths))
print("Min:", np.min(word_lengths))
print("Max:", np.max(word_lengths))

print("\nCharacter length:")
print("Mean:", np.mean(char_lengths))
print("Median:", np.median(char_lengths))
print("Min:", np.min(char_lengths))
print("Max:", np.max(char_lengths))

print("\nExamples:")
for label in [0, 1]:
    print(f"\nLabel {label} examples:")
    count = 0
    for item in data:
        if item["label"] == label:
            preview = item["text"][:300].replace("\n", " ")
            print("-", preview)
            count += 1
        if count >= 3:
            break