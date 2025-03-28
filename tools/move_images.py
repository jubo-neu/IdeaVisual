import os
import shutil


def move_images_to_folder(root_folder, target_folder='.../objaverse/basedata/depth_images'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    existing_files = set(os.listdir(target_folder))

    for folder in subfolders:
        png_files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]

        for file_name in png_files:
            old_file_path = os.path.join(folder, file_name)
            new_file_name = file_name

            counter = 1
            while new_file_name in existing_files:
                base, ext = os.path.splitext(file_name)
                new_file_name = f"{base}_{counter}{ext}"
                counter += 1

            new_file_path = os.path.join(target_folder, new_file_name)

            shutil.move(old_file_path, new_file_path)
            existing_files.add(new_file_name)
            print(f"Moved: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    root_folder = '.../objaverse/basedata/...'
    move_images_to_folder(root_folder)
    print("All images have been moved to the 'images' folder.")
