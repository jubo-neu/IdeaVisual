import os
import glob
from tqdm import tqdm


def keep_depth_pngs(directory):
    all_png_pattern = os.path.join(directory, '**', '*.png')
    all_png_files = glob.glob(all_png_pattern, recursive=True)

    depth_png_files = [f for f in all_png_files if 'depth' in os.path.basename(f).lower()]

    files_to_delete = [f for f in all_png_files if f not in depth_png_files]

    total_files = len(files_to_delete)
    print(f"total {total_files} delete files")

    for file_path in tqdm(files_to_delete, desc="process", unit="file"):
        try:
            os.remove(file_path)
            tqdm.write(f"deleted: {file_path}")
        except Exception as e:
            tqdm.write(f"not deleted {file_path}: {e}")


if __name__ == "__main__":
    root_directory = input("please enter your path：").strip()

    if not os.path.isdir(root_directory):
        print("invalid")
    else:
        keep_depth_pngs(root_directory)
