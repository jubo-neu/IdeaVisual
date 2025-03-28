import os
from PIL import Image


def rename_png_files_in_folder(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    png_files.sort()

    if not png_files:
        print("No PNG files found in the specified folder.")
        return

    for index, old_name in enumerate(png_files):
        new_name = f"{index:03d}.png"

        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(new_path):
            print(f"File {new_path} already exists. Skipping renaming of {old_name}.")
            continue

        try:
            with Image.open(old_path) as img:
                img.verify()
        except Exception as e:
            print(f"Skipping invalid image file: {old_name}, Error: {e}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed '{old_name}' to '{new_name}'")


if __name__ == "__main__":
    folder_path = '.../images'
    rename_png_files_in_folder(folder_path)
