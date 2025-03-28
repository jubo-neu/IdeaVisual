import os
from PIL import Image
import shutil

source_folder = '.../objaverse/Answerer_depth'
destination_folder = '.../objaverse/images'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

image_counter = 0

for i in range(1, 67):
    subfolder_name = f"000-{i:03d}"
    subfolder_path = os.path.join(source_folder, subfolder_name)

    if os.path.isdir(subfolder_path):
        for image_name in sorted(os.listdir(subfolder_path)):
            if image_name.lower().endswith('.png'):
                image_path = os.path.join(subfolder_path, image_name)

                try:
                    with Image.open(image_path) as img:
                        new_image_name = f"{image_counter:06d}.png"
                        new_image_path = os.path.join(destination_folder, new_image_name)

                        shutil.copyfile(image_path, new_image_path)
                        print(f"Processed {image_path} -> {new_image_path}")
                        image_counter += 1
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")

print("All images have been processed.")
