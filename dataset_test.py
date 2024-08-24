from IHC_Dataset import HE_IHC_Dataset
from SDUtils import SDUtils
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sdUtil = SDUtils(masterPath='../datasets/BCI_dataset')
train_he_dir, train_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TRAIN)
test_he_dir, test_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TEST)
# Create dataset instances
transform = transforms.Resize((512, 512))
train_dataset = HE_IHC_Dataset(train_he_dir, train_ihc_dir, transform=transform)
test_dataset = HE_IHC_Dataset(test_he_dir, test_ihc_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Training data set test')
print(f'Train dataset size: {len(train_dataset)}')
item = train_dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
print('Testing data set test')
print(f'test dataset size: {len(test_dataset)}')
item = test_dataset[123]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

i = 0
for item in tqdm(test_dataloader):
    control, ihc_img, prompt, image_name = item['hint'], item['jpg'], item['txt'], item['imgName']
    print(f'prompt: {prompt}')
    print(f'image_name: {image_name}')
    i += 1
    if i > 2:
        break