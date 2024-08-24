import os
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset

from logger import Logger

# Overriding the Dataset class so that it returns a tuple of corresponding HE and IHC images
class HE_IHC_Dataset(Dataset):  
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.logger = Logger().getLogger(name=__name__)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform
        self.he_images = os.listdir(he_dir)
        self.ihc_images = os.listdir(ihc_dir)
        self.ihc0_prompt = 'IHC score 0, No staining or incomplete membrane staining which is faint or barely perceptible in less than or equal to 10 percent of invasive tumor cells, Negative HER2 expression'
        self.ihc1_prompt = 'IHC score 1+, Incomplete membrane staining which is faint or barely perceptible in greater than 10 percent of invasive tumor cells, Low HER2 expression'
        self.ihc2_prompt = 'IHC score 2+, Weak to moderate membrane staining with uneven brownish yellow coloration in greater than 10 percent of invasive tumor cells, less than or equal to 10 percent of invasive tumor cells have circumferential membrane staining which is complete, intense, and has brownish coloration, Equivocal HER2 expression'
        self.ihc3_prompt = 'IHC score 3+, Greater than 10 percent of invasive tumor cells have circumferential membrane staining which is complete, intense, and has brownish coloration, Positive HER2 expression'

        # Ensure both directories have the same number of images and corresponding names
        assert len(self.he_images) == len(self.ihc_images), "Number of images in HE and IHC directories must match"
        self.he_images.sort()  # Sort for consistency
        self.ihc_images.sort()

    def __len__(self):
        return len(self.he_images)

    def __getitem__(self, idx):
        he_img_name = self.he_images[idx]
        ihc_img_name = self.ihc_images[idx]
        # self.logger.info(f'HE Image fetched: {he_img_name}')
        # self.logger.info(f'IHC Image fetched: {ihc_img_name}')

        ihc_score_prompt = ''
        pngIdx = he_img_name.index('.png')
        ihc_score = he_img_name[pngIdx-2:pngIdx]
        if ihc_score == '1+':
            ihc_score_prompt = self.ihc1_prompt
        elif ihc_score == '2+':
            ihc_score_prompt = self.ihc2_prompt
        elif ihc_score == '3+':
            ihc_score_prompt = self.ihc3_prompt
        else:
            ihc_score_prompt = self.ihc0_prompt

        he_img_path = os.path.join(self.he_dir, he_img_name)
        ihc_img_path = os.path.join(self.ihc_dir, ihc_img_name)
        # self.logger.info(f'HE Image path: {he_img_path}')
        # self.logger.info(f'IHC Image path: {ihc_img_path}')

        he_image = Image.open(he_img_path)
        ihc_image = Image.open(ihc_img_path)

        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)

        # converting to cv2 for compatibility with ControlNet scripts
        he_image = cv2.cvtColor(np.array(he_image), cv2.COLOR_BGR2RGB)
        ihc_image = cv2.cvtColor(np.array(ihc_image), cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        he_image = he_image.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        ihc_image = (ihc_image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=ihc_image, txt=ihc_score_prompt, hint=he_image)