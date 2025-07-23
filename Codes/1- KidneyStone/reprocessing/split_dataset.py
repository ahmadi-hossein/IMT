import os
import shutil
import random

input_dir = "C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/dataset"
output_dir = "C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/dataset_split"

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = os.listdir(input_dir)
    for cls in classes:
        class_path = os.path.join(input_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_end = int(len(images)*train_ratio)
        val_end = train_end + int(len(images)*val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split in splits:
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in splits[split]:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(split_dir, img)
                shutil.copy(src_path, dst_path)

split_dataset(input_dir, output_dir)
