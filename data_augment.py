import os
import cv2
import random
import numpy as np
from PIL import Image
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate,
    RandomBrightnessContrast, GaussianBlur, GaussNoise, HueSaturationValue
)
from albumentations.pytorch import ToTensorV2


def create_augmentation_pipeline():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=10, p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
    ])


def augment_image_and_label(image, label, transform):
    augmented = transform(image=image, mask=label)
    return augmented['image'], augmented['mask']


def load_image_label_pairs(img_folder, label_folder):
    img_files = sorted(os.listdir(img_folder))
    label_files = sorted(os.listdir(label_folder))

    pairs = [(os.path.join(img_folder, img), os.path.join(label_folder, lbl))
             for img, lbl in zip(img_files, label_files)]

    return pairs


def save_augmented_images(augmented_img, augmented_lbl, output_img_folder, output_lbl_folder, base_name, idx):
    img_name = f"{base_name}_aug_{idx}.png"
    lbl_name = f"{base_name}_aug_{idx}.png"

    cv2.imwrite(os.path.join(output_img_folder, img_name), augmented_img)
    cv2.imwrite(os.path.join(output_lbl_folder, lbl_name), augmented_lbl)


def augment_dataset(img_folder, label_folder, output_img_folder, output_lbl_folder, augmentations_per_image=5):
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_lbl_folder, exist_ok=True)

    augmentation_pipeline = create_augmentation_pipeline()
    pairs = load_image_label_pairs(img_folder, label_folder)

    for img_path, label_path in pairs:
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Load image and label as numpy arrays
        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        for idx in range(augmentations_per_image):
            augmented_img, augmented_lbl = augment_image_and_label(image, label, augmentation_pipeline)
            save_augmented_images(augmented_img, augmented_lbl, output_img_folder, output_lbl_folder, img_name, idx)

        print(f"Augmented {img_name} - {augmentations_per_image} times")


if __name__ == "__main__":
    img_folder = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\images"  # Replace with your image folder
    label_folder = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\labels"  # Replace with your label folder
    output_img_folder = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\images_op"  # Replace with your output image folder
    output_lbl_folder = r"D:\PyCharm\UNet\DataSetpreProcessing\Potsdam_images\labels_op"  # Replace with your output label folder

    augmentations_per_image = 5  # Number of augmentations per image

    augment_dataset(img_folder, label_folder, output_img_folder, output_lbl_folder, augmentations_per_image)
