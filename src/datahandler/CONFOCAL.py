import os

import cv2
import numpy as np
import torch

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class Confocal(DenoiseDataSet):
    '''
    Confocal dataset class for uint16 single-channel png images.
    All images are stored in a single folder.
    Supports mixed 8-bit and 16-bit images via per-image auto-detection.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = '../../../Dataset/T3/all_crop_pix512_pix1024_300pics'
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # Scan all PNG/TIF files in the dataset directory
        for file_name in sorted(os.listdir(dataset_path)):
            if file_name.lower().endswith(('.png', '.tif', '.tiff')):
                self.img_paths.append(os.path.join(dataset_path, file_name))

        assert len(self.img_paths) > 0, 'No images found in %s'%dataset_path
        print('Found %d confocal images in %s' % (len(self.img_paths), dataset_path))

    def _load_data(self, data_idx):
        img_path = self.img_paths[data_idx]

        # IMREAD_UNCHANGED preserves original bit depth
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, "failure on loading image - %s"%img_path

        # force single channel: strip extra dims, keep grayscale as-is
        if img.ndim > 2:
            img = img[:, :, 0]
        assert img.ndim == 2, \
            "unexpected image shape %s for %s" % (img.shape, img_path)

        # auto-detect bit depth: uint8 (max<=255) or uint16 (max>255)
        norm_factor = 255.0 if img.max() <= 255 else 65535.0

        # normalize to [0, 1]
        img = np.expand_dims(img.astype(np.float32), axis=0)
        img = img / norm_factor
        noisy_img = torch.from_numpy(np.ascontiguousarray(img))

        return {'real_noisy': noisy_img, 'norm_factor': norm_factor}


@regist_dataset
class prep_confocal(DenoiseDataSet):
    '''
    Dataset class for prepared confocal dataset which is cropped with overlap.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/confocal_s512_o0')

        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        img_path = os.path.join(self.dataset_path, 'RN', file_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, "failure on loading image - %s"%img_path

        # auto-detect bit depth: uint8 (max<=255) or uint16 (max>255)
        norm_factor = 255.0 if img.max() <= 255 else 65535.0

        # normalize to [0, 1]
        img = np.expand_dims(img.astype(np.float32), axis=0)
        img = img / norm_factor
        noisy_img = torch.from_numpy(np.ascontiguousarray(img))

        return {'real_noisy': noisy_img, 'norm_factor': norm_factor}
