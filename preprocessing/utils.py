import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.data import Dataset, DataLoader
from monai.data.image_reader import nib
from monai.transforms import CropForeground, Compose, LoadImaged, CropForegroundd, EnsureTyped


def read_data():
    # 读取数据并展示
    data = nib.load("../demo_data2/labelsTr/case_00002.nii.gz").get_fdata()
    print(data.shape)
    # 展示图片
    plt.imshow(data[100,:,:], cmap='gray')
    plt.show()


data = [{
    "image": "../demo_data2/imagesTr/case_00000_0000.nii.gz",  # CT图像的路径
    "label": "../demo_data2/labelsTr/case_00000.nii.gz"  # 标签图像的路径
}]


def crop_image_demo():
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        CropForegroundd(keys=["image", "label"], source_key="label", margin=0)  # 根据标签裁剪图像和标签
    ])
    transformed_data = transforms(data[0])
    # 展示transformed_data 和 原图像
    image = transformed_data["image"]
    label = transformed_data["label"]
    # 原图像
    load_transform = Compose([
        LoadImaged(keys=["image", "label"])
    ])
    # ------------------------------------
    original_data = load_transform(data[0])
    # 对比展示图
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0][0].imshow(original_data['image'][100], cmap='gray')
    axes[0][0].set_title('Original image')
    axes[0][1].imshow(transformed_data['image'][100], cmap='gray')
    axes[0][1].set_title('Cropped image')
    axes[1][0].imshow(original_data['label'][100])
    axes[1][0].set_title('Original label')
    axes[1][1].imshow(transformed_data['label'][100])
    axes[1][1].set_title('Cropped label')
    plt.show()

    # 计算原始图像的体素总数
    original_voxel_count = np.prod(original_data['image'].shape)
    # 计算裁剪后图像的体素总数
    cropped_voxel_count = np.prod(transformed_data['image'].shape)

    # 计算裁剪的体积百分比
    cropped_percentage = (cropped_voxel_count) / original_voxel_count

    print(f"Removed the area(percentage): {100 * (1 - cropped_percentage):.2f}%")

    # original shape vs cropped shape
    print(f"Original image shape: {original_data['image'].shape}")
    print(f"Cropped image shape: {transformed_data['image'].shape}")
    print(f"Original label shape: {original_data['label'].shape}")
    print(f"Cropped label shape: {transformed_data['label'].shape}")


if __name__ == "__main__":
    # read_data()
    crop_image_demo()

    # data = nib.load("../demo_data2/imagesTr/case_00000_0000.nii.gz")
    # # 查看体素间距
    # print(data.header.get_zooms())


