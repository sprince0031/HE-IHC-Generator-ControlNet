from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from IHC_Dataset import HE_IHC_Dataset
from SDUtils import SDUtils
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5 # <NOT IMPLEMENTED> slower learning rate for better results
sd_locked = True # <NOT IMPLEMENTED> Allowing the lower layers of the SD model to be retrained considering this is a specific case
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
sdUtil = SDUtils(masterPath='../datasets/BCI_dataset')
train_he_dir, train_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TRAIN)
test_he_dir, test_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TEST)
# Create dataset instances
transform = transforms.Resize((512, 512))
train_dataset = HE_IHC_Dataset(train_he_dir, train_ihc_dir, transform=transform)
test_dataset = HE_IHC_Dataset(test_he_dir, test_ihc_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(devices=1, accelerator='gpu', precision=16, max_epochs=50, callbacks=[logger])


# Train!
# if __name__ == '__main__':
trainer.fit(model, train_dataloader)
