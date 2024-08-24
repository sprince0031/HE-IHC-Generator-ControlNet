from PIL import Image
import matplotlib.pyplot as plt

# Class containing various utility functions for this notebook
class SDUtils():

  # Global variables
  TRAIN = 0
  TEST = 1
  VALIDATION = 2
  HE_dir_prefix = ""
  IHC_dir_prefix = ""

  def __init__(self, masterPath='./datasets/BCI_dataset'):
    # All dataset paths
    self.HE_dir_prefix = f'{masterPath}/HE'
    self.IHC_dir_prefix = f'{masterPath}/IHC'

  # util function to plot the original HE image, ground truth IHC image and transformed image from the diffuser model
  def plotImages(self, originalImageName, groundTruthImageName, ihcScore, dirType, transformedImage=None):

    if self.HE_dir_prefix == '' or self.IHC_dir_prefix == '':
      self.setDefaultDataPaths('')

    hePath, ihcPath = self.getDataPathFromType(dirType)
    originalImage = Image.open(f'{hePath}/{originalImageName}')
    groundTruthImage = Image.open(f'{ihcPath}/{groundTruthImageName}')
    imageDict = {'H&E Image (reference)': originalImage,
                 f'IHC Image (ground truth), IHC score: {ihcScore}': groundTruthImage}

    if transformedImage is not None:
      imageDict['Generated IHC Image'] = transformedImage
      plt.figure(figsize=(15, 5))
    else:
      plt.figure(figsize=(10, 5))

    self.addImageToPlot(imageDict)
    plt.show()

  # Add image to plot
  def addImageToPlot(self, imageDict):
    numOfImages = len(imageDict)
    for i, (plotName, image) in enumerate(imageDict.items()):
      plt.subplot(1, numOfImages, i+1)
      plt.axis('off')
      plt.imshow(image)
      plt.title(plotName)

  # Plot metrics
  def plotMetrics(self, metrics):
    numOfMetrics = len(metrics)
    plt.figure(figsize=(numOfMetrics * 5, 5))
    for i, (plotName, metric) in enumerate(metrics.items()):
      plt.subplot(1, numOfMetrics, i+1)
      plt.plot(metric['metric'])
      plt.xlabel(metric['xlabel'])
      plt.ylabel(metric['ylabel'])
      plt.title(plotName)

  # Returns the folder path for HE and IHC images based on the type of data requested
  def getDataPathFromType(self, dirType):
    if dirType == self.TRAIN:
      subfolder = 'train'
    elif dirType == self.TEST:
      subfolder = 'test'
    elif dirType == self.VALIDATION:
      subfolder = 'validation'

    return f'{self.HE_dir_prefix}/{subfolder}', f'{self.IHC_dir_prefix}/{subfolder}'

  # Convert to 5 digit number i.e., pad with leading zeroes if necessary
  # This is useful because of the naming convention of image files in the BCI dataset
  def padZeroes(self, number, numDigits=5):
    return str(number).zfill(numDigits)
