from torchvision import transforms
from PIL import Image
import os
import shutil

train_dir = r"C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/dataset_split/train"
augmented_dir = r"C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/augmentation_train"

# Resize ثابت
resize_transform = transforms.Resize((512, 512))

# Augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
])

os.makedirs(augmented_dir, exist_ok=True)

classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# =========================================
#  مرحله 1: کپی تصاویر اصلی به augmented_dir
# =========================================
for cls in classes:
    class_input_path = os.path.join(train_dir, cls)
    class_output_path = os.path.join(augmented_dir, cls)
    os.makedirs(class_output_path, exist_ok=True)

    for img_file in os.listdir(class_input_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(class_input_path, img_file)
            dst = os.path.join(class_output_path, img_file)
            shutil.copy(src, dst)  # 👈 اضافه شده

# =========================================
#  مرحله 2: تولید نسخه‌های افزایشی (Augmented)
# =========================================
for cls in classes:
    class_input_path = os.path.join(train_dir, cls)
    class_output_path = os.path.join(augmented_dir, cls)

    images = [img for img in os.listdir(class_input_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Processing class '{cls}' with {len(images)} images...")

    for idx, img_file in enumerate(images):
        img_path = os.path.join(class_input_path, img_file)
        image = Image.open(img_path).convert("RGB")

        # Resize ثابت 512x512
        image = resize_transform(image)

        for i in range(3):
            augmented_img = augmentation(image)
            save_path = os.path.join(class_output_path, f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg")
            augmented_img.save(save_path)

print(" Augmentation & Copying Original Images Done.")









# ============ مسیرها ============
train_dir = r"C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/augmentation_train"
val_dir =  r"C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/dataset_split/val"
test_dir =  r"C:/Users/pc/Documents/project machin learning/IMT/Datasets/1- KideyStone/dataset_split/test"

# ============ ترنسفورم‌ها ============
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# ============ لود کردن دیتاست‌ها ============
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}

dataLoaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

class_name = image_datasets['train'].classes
print(f"Classes: {class_name}")
