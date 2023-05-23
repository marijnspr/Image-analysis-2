import os
import shutil

def move_images_to_class_directories(source_dir, classes):
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    images = os.listdir(source_dir)
    for image_name in images:
        if not image_name.endswith('.jpg'):
            continue

        class_name = image_name.split("_")[0]
        if class_name in classes:
            src_path = os.path.join(source_dir, image_name)
            dest_dir = os.path.join(source_dir, class_name)
            dest_path = os.path.join(dest_dir, image_name)
            shutil.move(src_path, dest_path)
            print(f"Moved {image_name} to {dest_path}")

# Specify the source directory containing the images
source_directory = "foto/validation"

# Specify the class names
classes = ["n02096051", "n02107683", "n02110185"]

# Move the corresponding images to their respective class directories
move_images_to_class_directories(source_directory, classes)
