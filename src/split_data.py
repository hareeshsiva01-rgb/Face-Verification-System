import csv
import random
import os


INPUT_FILE = "dataset/pairs.csv"

TRAIN_FILE = "dataset/train_pairs.csv"
VAL_FILE = "dataset/val_pairs.csv"

TRAIN_RATIO = 0.8


# ----------------------------------------
# Load pairs
# ----------------------------------------

pairs = []

with open(INPUT_FILE, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        pairs.append([
            row["image1"],
            row["image2"],
            row["label"]
        ])


print()
print("Total pairs:", len(pairs))


# ----------------------------------------
# Shuffle pairs
# ----------------------------------------

random.seed(42)

random.shuffle(pairs)


# ----------------------------------------
# Calculate split
# ----------------------------------------

train_size = int(
    len(pairs) * TRAIN_RATIO
)

train_pairs = pairs[:train_size]

val_pairs = pairs[train_size:]


# ----------------------------------------
# Create output folder
# ----------------------------------------

os.makedirs(
    "dataset",
    exist_ok=True
)


# ----------------------------------------
# Save training pairs
# ----------------------------------------

with open(
    TRAIN_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image1",
        "image2",
        "label"
    ])

    writer.writerows(train_pairs)


# ----------------------------------------
# Save validation pairs
# ----------------------------------------

with open(
    VAL_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image1",
        "image2",
        "label"
    ])

    writer.writerows(val_pairs)


# ----------------------------------------
# Display result
# ----------------------------------------

print()
print("Dataset split completed.")

print(
    "Training pairs:",
    len(train_pairs)
)

print(
    "Validation pairs:",
    len(val_pairs)
)

print()
print("Training file:")
print(TRAIN_FILE)

print()
print("Validation file:")
print(VAL_FILE)
