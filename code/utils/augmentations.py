#python
import random
#libs
from PIL import Image
#pytorch
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

def augmentation(image, mask):
    
    hflip = True and random.random() < 0.5
    vflip = True and random.random() < 0.5
    rot = True and random.random() < 0.5

    def _augment(img, mask):
        if hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if rot:
            angle = torch.randint(-45, 45, (1, )).item()
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        return img, mask

    return _augment(image, mask) 

def transform_data(image, mask, augment=True):
    
    resize = transforms.Resize(size=(418, 418), interpolation=Image.NEAREST)
    image = resize(image)
    mask = resize(mask)
    
    if augment == True:
        image, mask = augmentation(image, mask)
    
    #Random crop
    # params = transforms.RandomCrop.get_params(
    #     image, output_size=(256, 256))
    # image = TF.crop(image, *params)
    # mask = TF.crop(mask, *params)
    return image, mask
