from share import *

from cldm.model import create_model, load_state_dict
import cv2
from annotator.util import resize_image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from IHC_Dataset import HE_IHC_Dataset
from SDUtils import SDUtils
from tqdm import tqdm
from pytorch_lightning import seed_everything

# Configs
resume_path = './lightning_logs/version_7/checkpoints/epoch=49-step=48699.ckpt' # checkpoint path
batch_size = 1
ddim_steps = 50


model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
num_samples = 1
seed = 42

sdUtil = SDUtils(masterPath='../datasets/BCI_dataset')
test_he_dir, test_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TEST)
transform = transforms.Resize((512, 512))
test_dataset = HE_IHC_Dataset(test_he_dir, test_ihc_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for item in tqdm(test_dataloader):
    with torch.no_grad():
        control, ihc_img, prompt, image_name = item['hint'], item['jpg'], item['txt'], item['imgName']
        save_path = './image_log/test/no_prompt_v2/'

        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # seed_everything(seed)

        control = control.cuda()
        # c = model.get_unconditional_conditioning(batch_size)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(['']) * num_samples]}
        uc_cross = model.get_unconditional_conditioning(batch_size)
        uc_full = {"c_concat": [control], "c_crossattn": [uc_cross]}
        b, c, h, w = cond["c_concat"][0].shape
        shape = (4, h // 8, w // 8)

        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, 
                                                    shape, cond, verbose=False, eta=0.0, 
                                                    unconditional_guidance_scale=9.0,
                                                    unconditional_conditioning=uc_full
                                                    )
        x_samples = model.decode_first_stage(samples)
        # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples = x_samples.squeeze(0)
        x_samples = (x_samples + 1.0) / 2.0
        x_samples = x_samples.transpose(0, 1).transpose(1, 2)
        x_samples = x_samples.cpu().numpy()
        x_samples = (x_samples * 255).astype(np.uint8)

        Image.fromarray(x_samples).save(save_path + image_name[0])
