"""
1.
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging

2. def convert_kits2019(kits_base_dir: str, raw_output_root)

3. rm -rf /path/to/directory/{,.[!.],..?}*

4. bounding_box

5. (TODO: 把val和test移过去)




"""
import shutil
from os.path import join

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subdirs


def convert_kits2019(kits_base_dir: str, raw_output_root):
    task_name = "KiTS2019"

    foldername = "Dataset_%s" % (task_name)

    # setting up nnU-Net folders
    out_base = join(raw_output_root, foldername)
    imagestr = join(out_base, "imagesTr")  # images(Train)
    # 在这下面创建Train和Val文件夹
    train_imagestr = join(imagestr, "Train")
    val_imagestr = join(imagestr, "Val")

    labelstr = join(out_base, "labelsTr")  # labels(Train)
    train_labelstr = join(labelstr, "Train")
    val_labelstr = join(labelstr, "Val")

    imagesTs = join(out_base, "imagesTs")  # images(Test)

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagesTs)

    cases = subdirs(kits_base_dir, prefix='case_', join=False)
    for tr in cases:
        case_id = int(tr.split('_')[-1])
        if case_id < 210:
            if case_id == 160:  # 忽略case_160
                pass
            else:
                # 0-178: Train, 179-209: Test
                if case_id < 179:
                    shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(train_imagestr, f'{tr}_0000.nii.gz'))
                    shutil.copy(join(kits_base_dir, tr, 'segmentation.nii.gz'), join(train_labelstr, f'{tr}.nii.gz'))
                else:
                    shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(val_imagestr, f'{tr}_0000.nii.gz'))
                    shutil.copy(join(kits_base_dir, tr, 'segmentation.nii.gz'), join(val_labelstr, f'{tr}.nii.gz'))
        else:
            shutil.copy(join(kits_base_dir, tr, 'imaging.nii.gz'), join(imagesTs, f'{tr}_0000.nii.gz'))


