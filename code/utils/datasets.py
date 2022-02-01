#python

#libs
from PIL import Image
import numpy as np

#pytorch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

###
from .augmentations import transform_data

###
class SegDataset(Dataset):
    def __init__(self, files_name, train=True):
        self.files_name = files_name
        self.train = train

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        image_path = self.files_name[index]

        stem = image_path.stem
        image_path = str(image_path)
        mask_path = image_path.replace('image', 'label').replace(stem, f'p{stem}')

        #open image
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        assert image.size == mask.size, f'Image and mask should be of same size, but are {image.size} and {mask.size}'

        if self.train:
            image, mask = transform_data(image, mask, augment=True)
        
        image = transforms.ToTensor()(image)        
        mask = np.array(mask) / 255
        mask = torch.from_numpy(mask)

        return {
            'image': image.type(torch.FloatTensor),
            'mask': mask.type(torch.long),
            'idx' : image_path.split('/')[-1].split('.')[0]
        }

        