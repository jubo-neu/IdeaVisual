import os
import glob
from tqdm import tqdm


def delete_depth_pngs(directory):
    pattern = os.path.join(directory, '**', '*_mm*.png')
    files_to_delete = glob.glob(pattern, recursive=True)

    if not files_to_delete:
        print("no delete files")
        return

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
        delete_depth_pngs(root_directory)
