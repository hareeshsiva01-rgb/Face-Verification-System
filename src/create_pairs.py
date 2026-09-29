import os
import csv

DATASET_PATH = "dataset/raw"
OUTPUT_FILE = "dataset/pairs.csv"

pairs = []

# Get all person folders
people = [
    person
    for person in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, person))
]

print("People found:", people)


# --------------------------------
# SAME PERSON PAIRS
# --------------------------------

for person in people:

    person_folder = os.path.join(DATASET_PATH, person)

    images = [
        image
        for image in os.listdir(person_folder)
        if image.endswith(".jpg")
    ]

    for i in range(len(images)):

        for j in range(i + 1, len(images)):

            image1 = os.path.join(person_folder, images[i])
            image2 = os.path.join(person_folder, images[j])

            pairs.append([
                image1,
                image2,
                1
            ])


# --------------------------------
# DIFFERENT PERSON PAIRS
# --------------------------------

for i in range(len(people)):

    for j in range(i + 1, len(people)):

        person1 = people[i]
        person2 = people[j]

        folder1 = os.path.join(DATASET_PATH, person1)
        folder2 = os.path.join(DATASET_PATH, person2)

        images1 = [
            image
            for image in os.listdir(folder1)
            if image.endswith(".jpg")
        ]

        images2 = [
            image
            for image in os.listdir(folder2)
            if image.endswith(".jpg")
        ]

        for image1 in images1:

            for image2 in images2:

                path1 = os.path.join(folder1, image1)
                path2 = os.path.join(folder2, image2)

                pairs.append([
                    path1,
                    path2,
                    0
                ])


# --------------------------------
# SAVE CSV
# --------------------------------

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "image1",
        "image2",
        "label"
    ])

    writer.writerows(pairs)


# --------------------------------
# DISPLAY RESULT
# --------------------------------

same_count = sum(1 for pair in pairs if pair[2] == 1)
different_count = sum(1 for pair in pairs if pair[2] == 0)

print()
print("Pair creation completed.")
print("Same-person pairs:", same_count)
print("Different-person pairs:", different_count)
print("Total pairs:", len(pairs))
print()
print("Saved to:", OUTPUT_FILE)