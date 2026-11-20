
from pathlib import Path
from collections import defaultdict
import csv
import random
import re
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Projects\Wildfire_Smoke_Project")

IMAGES_DIR = PROJECT_DIR / "smoke_images"
LABELS_DIR = PROJECT_DIR / "smoke_labels"

OUTPUT_DIR = PROJECT_DIR / "datasets" / "experiment_5"

TRAIN_RATIO = 0.80
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Expected format: timestamp_+offset.jpg or timestamp_-offset.jpg
FILENAME_PATTERN = re.compile(r"^(\d+)_([+-]\d+)$")


# ============================================================
# IDENTIFY RECORDING
# ============================================================

def get_recording_id(image_path):
    """Recover the recording/event timestamp from a FIgLib filename."""
    match = FILENAME_PATTERN.fullmatch(image_path.stem)

    if not match:
        raise ValueError(f"Unexpected filename: {image_path.name}")

    timestamp = int(match.group(1))
    offset = int(match.group(2))

    return timestamp - offset


# ============================================================
# VALIDATE ANNOTATIONS
# ============================================================

def validate_label(label_path):
    """Check YOLO class IDs and normalized bounding boxes."""
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "Expected 5 values."
            )

        if values[0] != "0":
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "Expected class ID 0."
            )

        try:
            x, y, w, h = map(float, values[1:])
        except ValueError:
            raise ValueError(f"Invalid coordinates: {label_path.name}")

        if not (
            0 <= x <= 1 and
            0 <= y <= 1 and
            0 < w <= 1 and
            0 < h <= 1 and
            x - w / 2 >= -1e-6 and
            x + w / 2 <= 1 + 1e-6 and
            y - h / 2 >= -1e-6 and
            y + h / 2 <= 1 + 1e-6
        ):
            raise ValueError(
                f"{label_path.name}, line {line_number}: "
                "Bounding box outside image bounds."
            )

    return bool(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError("Missing smoke_images or smoke_labels folder.")

    images = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    labels = {
        path.stem: path
        for path in LABELS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    # Organize images into groups based on recording ID.
    recordings = defaultdict(list)
    seen_stems = set()
    positive_count = 0
    negative_count = 0

    for image in images:
        if image.stem in seen_stems:
            raise ValueError(f"Duplicate image identifier: {image.stem}")
        seen_stems.add(image.stem)

        if image.stem not in labels:
            raise ValueError(f"Missing annotation: {image.name}")

        is_positive = validate_label(labels[image.stem])

        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        recording_id = get_recording_id(image)
        recordings[recording_id].append((image, labels[image.stem], is_positive))

    orphaned = set(labels) - seen_stems
    if orphaned:
        raise ValueError(f"Orphaned labels: {sorted(orphaned)[:10]}")

    print("DATASET VALIDATION")
    print("-" * 45)
    print(f"Images:          {len(images)}")
    print(f"Smoke-positive:  {positive_count}")
    print(f"Smoke-negative:  {negative_count}")
    print(f"Recordings:      {len(recordings)}")

    if len(recordings) < 2:
        raise ValueError("At least two recordings are needed for a split.")

    # Randomize recording groups, not individual frames.
    recording_ids = sorted(recordings)
    rng = random.Random(SEED)
    rng.shuffle(recording_ids)

    # Allocate groups until approximately 80% of images are training.
    target_train = round(len(images) * TRAIN_RATIO)
    train_ids = set()
    train_count = 0

    for recording_id in recording_ids:
        if train_count < target_train:
            train_ids.add(recording_id)
            train_count += len(recordings[recording_id])

    val_ids = set(recording_ids) - train_ids

    if not val_ids:
        raise ValueError("No validation recordings were allocated.")

    # Refuse to overwrite an existing experiment.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Rename or remove it before running this script."
        )

    # Create YOLO directory structure.
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True)

    manifest = []
    split_counts = {"train": 0, "val": 0}

    # Copy images and their matching labels.
    for recording_id in sorted(recordings):
        split = "train" if recording_id in train_ids else "val"

        for image, label, is_positive in recordings[recording_id]:
            shutil.copy2(image, OUTPUT_DIR / "images" / split / image.name)
            shutil.copy2(label, OUTPUT_DIR / "labels" / split / label.name)

            split_counts[split] += 1

            manifest.append({
                "filename": image.name,
                "recording_id": recording_id,
                "split": split,
                "smoke_positive": is_positive,
            })

    # Write YOLO configuration.
    yaml_text = (
        f"path: {OUTPUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: smoke\n"
    )

    (OUTPUT_DIR / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    # Save the recording assignments for reproducibility.
    with (OUTPUT_DIR / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "recording_id", "split", "smoke_positive"]
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\nEXPERIMENT 5 DATASET CREATED")
    print("-" * 45)
    print(f"Training images:    {split_counts['train']}")
    print(f"Validation images:  {split_counts['val']}")
    print(f"Training recordings: {len(train_ids)}")
    print(f"Validation recordings: {len(val_ids)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
