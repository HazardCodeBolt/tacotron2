"""
Upload dataset_new_omani_processed/ to Google Drive (Colab environment).

Run this in a Colab notebook cell:
    !python upload_preprocessed_to_drive.py [--local DIR] [--drive-folder NAME]

Or import and call upload() directly from a notebook cell.
"""

import argparse
import sys
from pathlib import Path


DEFAULTS = dict(
    local_dir    = "dataset_new_omani_processed",
    drive_folder = "dataset_new_omani_processed",
)


def _mount_drive(mount_point="/content/drive"):
    """Mount Google Drive if not already mounted."""
    from google.colab import drive
    if not Path(mount_point).exists() or not any(Path(mount_point).iterdir()):
        print("Mounting Google Drive ...")
        drive.mount(mount_point, force_remount=False)
    else:
        print("Google Drive already mounted.")
    return Path(mount_point)


def _resolve_local_root():
    """Find the project root (directory containing commons/dataset.py)."""
    root = Path(__file__).resolve().parent
    for _ in range(4):
        if (root / "commons" / "dataset.py").exists():
            return root
        root = root.parent
    return Path(__file__).resolve().parent


def upload(local_dir=None, drive_folder=None, mount_point="/content/drive"):
    """
    Upload local_dir to My Drive/<drive_folder>/ on the connected Google Drive.

    Args:
        local_dir:    Local path to the preprocessed dataset folder.
        drive_folder: Destination folder name inside Google Drive 'My Drive'.
        mount_point:  Where Colab mounts Drive (default /content/drive).
    """
    import shutil

    local_dir    = Path(local_dir    or (_resolve_local_root() / DEFAULTS["local_dir"]))
    drive_folder = drive_folder or DEFAULTS["drive_folder"]

    if not local_dir.exists():
        print(f"ERROR: local directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    drive_root = _mount_drive(mount_point)
    my_drive   = drive_root / "MyDrive"
    dest       = my_drive / drive_folder

    files = sorted(local_dir.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    print(f"\nLocal  : {local_dir}  ({len(file_list)} files)")
    print(f"Dest   : {dest}\n")

    ok = skipped = errors = 0
    total = len(file_list)

    for i, src_file in enumerate(file_list, 1):
        rel      = src_file.relative_to(local_dir)
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
            print(f"[{i:4d}/{total}] -> {rel}  skipped (same size)")
            skipped += 1
            continue

        try:
            shutil.copy2(src_file, dst_file)
            print(f"[{i:4d}/{total}] ok {rel}")
            ok += 1
        except Exception as e:
            print(f"[{i:4d}/{total}] ERROR {rel}  {e}")
            errors += 1

    print(f"\nDone — uploaded={ok}  skipped={skipped}  errors={errors}")
    print(f"Drive path: {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload preprocessed Omani dataset to Google Drive (Colab)"
    )
    parser.add_argument(
        "--local",
        default=None,
        help=f"Local folder to upload (default: <project_root>/{DEFAULTS['local_dir']})",
    )
    parser.add_argument(
        "--drive-folder",
        default=DEFAULTS["drive_folder"],
        help=f"Destination folder name in My Drive (default: {DEFAULTS['drive_folder']})",
    )
    parser.add_argument(
        "--mount-point",
        default="/content/drive",
        help="Colab Drive mount point (default: /content/drive)",
    )
    args = parser.parse_args()
    upload(
        local_dir    = args.local,
        drive_folder = args.drive_folder,
        mount_point  = args.mount_point,
    )


if __name__ == "__main__":
    main()
