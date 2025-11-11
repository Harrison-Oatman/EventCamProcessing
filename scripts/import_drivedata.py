"""
import_drivedata.py

- Downloads all `.raw` files from specified google drive folder
    and adds them to data/raw_files/.
- Skips files that already exist locally (only downloads new .raw files).

**Note: I had to make the drive folder
    publically available (with a link) to make this work

To run this script:
    uv add gdown  (first time running)
    uv run scripts/import_drivedata.py
"""

from pathlib import Path

import gdown

# URL ID for the folder (after drive/folders/)
FOLDER_ID = "1uQv9_vtb6A2357m40pa7H9tvaufURm0l"

# Local destination directory
LOCAL_DIR = Path("data/raw_files")


def download_raw_files(folder_id: str, dest: Path):
    """download new .raw files from google drive folder"""
    dest.mkdir(parents=True, exist_ok=True)

    print(f"importing .raw files from drive folder: {folder_id}")
    print(f"local data-file directory: {dest.resolve()}")

    # gdown downloads all contents of the folder, so we'll put them in
    # a temporary directory
    temp_dir = dest / "_temp_gdown"
    temp_dir.mkdir(exist_ok=True)

    # Download folder contents
    gdown.download_folder(
        id=folder_id, output=str(temp_dir), quiet=False, use_cookies=False
    )

    # Move only .raw files that are new to data/raw_files
    count = 0
    for file in temp_dir.glob("**/*.raw"):
        target = dest / file.name
        if not target.exists():
            file.replace(target)
            count += 1
            print(f"Downloaded: {file.name}")
        else:
            print(f"Skipped existing file: {file.name}")

    # Get rid of the temporary directory
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()

    if count == 0:
        print("All .raw files are already up to date!")
    else:
        print(f"Downloaded {count} new files!")


if __name__ == "__main__":
    download_raw_files(FOLDER_ID, LOCAL_DIR)
