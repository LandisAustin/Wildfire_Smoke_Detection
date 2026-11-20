from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

project_dir = Path(r"D:\Projects\Wildfire_Smoke_Project")

images_dir = project_dir / "smoke_images"
labels_dir = project_dir / "smoke_labels"

# Folders for files that need attention
missing_dir = project_dir / "images_needing_annotation"
orphaned_dir = project_dir / "orphaned_labels"

image_extensions = {".jpg", ".jpeg", ".png"}


# ============================================================
# ORGANIZE UNMATCHED FILES
# ============================================================

def main():
    # Verify the source folders exist
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Could not find smoke_images or smoke_labels.")

    # Find all images and labels
    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    labels = sorted(
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    # Identify files without matching counterparts
    missing_images = [
        p for p in images if p.stem not in label_stems
    ]

    orphaned_labels = [
        p for p in labels if p.stem not in image_stems
    ]

    # Create folders for unmatched files
    missing_dir.mkdir(exist_ok=True)
    orphaned_dir.mkdir(exist_ok=True)

    # Move images that still need annotations
    for image in missing_images:
        destination = missing_dir / image.name

        # Avoid accidentally overwriting an existing file
        if destination.exists():
            print(f"Skipping existing destination: {destination.name}")
            continue

        shutil.move(str(image), str(destination))
        print(f"Moved image: {image.name}")

    # Move labels without corresponding images
    for label in orphaned_labels:
        destination = orphaned_dir / label.name

        if destination.exists():
            print(f"Skipping existing destination: {destination.name}")
            continue

        shutil.move(str(label), str(destination))
        print(f"Moved orphaned label: {label.name}")

    # Display results
    print("\nDATASET CLEANUP COMPLETE")
    print("-" * 45)
    print(f"Images needing annotation: {len(missing_images)}")
    print(f"Orphaned labels identified: {len(orphaned_labels)}")
    print(f"Missing images folder:     {missing_dir}")
    print(f"Orphaned labels folder:    {orphaned_dir}")


if __name__ == "__main__":
    main()