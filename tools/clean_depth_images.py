import os


def get_category_number(filename):
    prefix = filename.split('_view')[0]
    numbers = ''.join(filter(str.isdigit, prefix))
    return int(numbers) if numbers else None


def clean_depth_images(rgb_folder, depth_folder):
    rgb_categories = set()
    for file_name in os.listdir(rgb_folder):
        category_number = get_category_number(file_name)
        if category_number is not None:
            rgb_categories.add(category_number)

    for file_name in os.listdir(depth_folder):
        category_number = get_category_number(file_name)
        if category_number not in rgb_categories:
            file_path = os.path.join(depth_folder, file_name)
            print(f"Deleting {file_path}")
            os.remove(file_path)


rgb_folder = '.../objaverse/Answerer/images'
depth_folder = '.../objaverse/basedata/depth_images'
clean_depth_images(rgb_folder, depth_folder)
