import os
import re
from pathlib import Path
import sys

def rename_feature_files(directory_path: str):
    """
    Scans a directory for pickle files with single-digit video numbers
    (e.g., "video_5_25.0fps.pkl") and renames them with zero-padding
    (e.g., "video_05_25.0fps.pkl").
    """
    target_dir = Path(directory_path)
    if not target_dir.is_dir():
        print(f"Error: Directory not found at '{target_dir}'")
        return

    print(f"Scanning for files to rename in: {target_dir}")

    # This pattern finds filenames like "video_N_..." where N is a single digit.
    pattern = re.compile(r"^(video_)(\d)_(.*\.pkl)$")

    files_renamed_count = 0
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        if match:
            # Construct the new filename with zero-padding, e.g., "video_05_..."
            new_filename = f"{match.group(1)}0{match.group(2)}_{match.group(3)}"

            old_filepath = target_dir / filename
            new_filepath = target_dir / new_filename

            try:
                os.rename(old_filepath, new_filepath)
                print(f"  Renamed: '{filename}' -> '{new_filename}'")
                files_renamed_count += 1
            except OSError as e:
                print(f"Error renaming file {filename}: {e}")

    if files_renamed_count == 0:
        print("No files needed renaming.")
    else:
        print(f"\nSuccessfully renamed {files_renamed_count} files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_pkl_files.py <path_to_your_fps_folder>")
        print("Example: python rename_pkl_files.py /home/jovyan/Adv.CV/.../cholec80_pickle_export/25.0fps")
        sys.exit(1)

    path_to_scan = sys.argv[1]
    rename_feature_files(path_to_scan)