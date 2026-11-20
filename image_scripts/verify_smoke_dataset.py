from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

project_dir = Path(r"D:\Projects\Wildfire_Smoke_Project")

images_dir = project_dir / "smoke_images"
labels_dir = project_dir / "smoke_labels"

image_extensions = {".jpg", ".jpeg", ".png"}


# ============================================================
# VERIFY DATASET
# ============================================================

def main():
    # Verify both folders exist
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    # Collect all image and label files
    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    labels = sorted(labels_dir.rglob("*.txt"))

    # Match images and labels using their filename stems
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    # Identify missing and extra labels
    missing_labels = sorted(image_stems - label_stems)
    orphaned_labels = sorted(label_stems - image_stems)

    # Count smoke-positive and smoke-negative images
    positive_count = 0
    negative_count = 0

    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"

        if not label_path.exists():
            continue

        if label_path.read_text(encoding="utf-8").strip():
            positive_count += 1
        else:
            negative_count += 1

    # Print dataset summary
    print("\nSMOKE DATASET VERIFICATION")
    print("-" * 45)
    print(f"Total images:         {len(images)}")
    print(f"Total labels:         {len(labels)}")
    print(f"Smoke-positive:       {positive_count}")
    print(f"Smoke-negative:       {negative_count}")
    print(f"Missing labels:       {len(missing_labels)}")
    print(f"Orphaned labels:      {len(orphaned_labels)}")

    # Display missing labels
    if missing_labels:
        print("\nImages missing labels:")
        for stem in missing_labels[:20]:
            print(f"  {stem}")

    # Display labels without images
    if orphaned_labels:
        print("\nLabels without images:")
        for stem in orphaned_labels[:20]:
            print(f"  {stem}")

    if not missing_labels and not orphaned_labels:
        print("\nAll images and labels are matched!")
    else:
        print("\nDataset needs attention before training.")


if __name__ == "__main__":
    main()