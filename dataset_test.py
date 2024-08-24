from IHC_Dataset import HE_IHC_Dataset
from SDUtils import SDUtils
from torchvision import transforms

sdUtil = SDUtils(masterPath='../datasets/BCI_dataset')
train_he_dir, train_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TRAIN)
test_he_dir, test_ihc_dir = sdUtil.getDataPathFromType(SDUtils.TEST)
# Create dataset instances
transform = transforms.Resize((512, 512))
train_dataset = HE_IHC_Dataset(train_he_dir, train_ihc_dir, transform=transform)
test_dataset = HE_IHC_Dataset(test_he_dir, test_ihc_dir, transform=transform)
print(len(train_dataset))
# print(len(dataset))
item = train_dataset[1234]
# item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
