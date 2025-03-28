import os


def rename_images_in_folders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    subfolders.sort()

    for index, folder in enumerate(subfolders):
        prefix = f"{index}_"

        png_files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

        for file_name in png_files:
            old_file_path = os.path.join(folder, file_name)
            new_file_name = prefix + file_name
            new_file_path = os.path.join(folder, new_file_name)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    root_folder = '.../objaverse/basedata/...'
    rename_images_in_folders(root_folder)
    print("All images have been renamed.")
