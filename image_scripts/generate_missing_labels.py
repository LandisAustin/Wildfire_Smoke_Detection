from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

project_dir = Path(r"D:\Projects\Wildfire_Smoke_Project")

images_dir = project_dir / "smoke_images"
labels_dir = project_dir / "smoke_labels"

image_extensions = {".jpg", ".jpeg", ".png"}


# ============================================================
# GENERATE MISSING LABELS
# ============================================================

def main():
    # Verify both directories exist
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    created = 0
    existing = 0

    # Search for images without matching annotation files
    for image in sorted(images_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / f"{image.stem}.txt"

        if label_path.exists():
            existing += 1
            continue

        # Create an empty YOLO annotation file (no smoke)
        label_path.touch(exist_ok=False)
        created += 1

        print(f"Created: {label_path.name}")

    # Display summary
    print("\nLABEL GENERATION COMPLETE")
    print("-" * 45)
    print(f"Existing labels: {existing}")
    print(f"Empty labels created: {created}")
    print(f"Total matched: {existing + created}")


if __name__ == "__main__":
    main()