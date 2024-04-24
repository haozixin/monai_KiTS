import glob
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm


def check_data_shape(path, data_type='imaging'):
    """
    读取数据，统计数据的形状：最大值、最小值、均值、方差等
    :param path: 数据的路径根目录
    :return:
    """
    # 遍历数据文件夹找到所有imaging.nii.gz文件
    nii_files = glob.glob(os.path.join(path, '**', data_type+'.nii.gz'), recursive=True)

    shapes = []
    case = 0
    for file in tqdm(nii_files):
        # 读取数据nii
        data = nib.load(file).get_fdata()
        shapes.append(data.shape)
        case += 1

    # 统计数据形状
    shapes = np.array(shapes)
    print(f"数据形状的最大值：{shapes.max(axis=0)}")
    print(f"数据形状的最小值：{shapes.min(axis=0)}")
    print(f"数据形状的均值：{shapes.mean(axis=0)}")
    print(f"数据形状的方差：{shapes.var(axis=0)}")

    """ 
    cases 0-167  (since only case 160 is not 512*512, so I discard it.)
    数据形状的最大值：[1059,  512,  512]
    数据形状的最小值：[ 29, 512, 512]
    数据形状的均值：[230.71856287, 512,    512]
    数据形状的方差：[51427.1842662,  0,    0]
    
    case 168-209
    数据形状的最大值：[620, 512, 512]
    数据形状的最小值：[ 60, 512, 512]
    数据形状的均值：[158.14285714,  512,  512]
    数据形状的方差：[12027.50340136,  0,  0 ]
    
    
    """







if __name__=="__main__":
    # check_data_shape("F:\myUnet_data\\train", data_type='imaging')

    data = nib.load("E:/4Melbourne_uni_2024_S1/research/myNet/demo_data/train/case_00000/segmentation.nii.gz").get_fdata()

    # 所有的标签值
    print(np.unique(data))
