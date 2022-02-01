#
import os
#
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

#
import torch
import torch.nn as nn
from torchvision import transforms
# from torchvision.utils import make_grid

#
from utils.loss import celoss
from utils.metrics import compute_mean_iou
from utils.util import mkdir
#
def eval_net(encoder, decoder, loader, device, current_step, wandb, save_dir, last=False):    
    encoder.eval()
    decoder.eval()
    loss = 0
    miou_list = []
    tensor_list = []
    
    for batch in tqdm(loader):
        imgs, mask_gt, idx = batch['image'], batch['mask'], batch['idx'][0]

        imgs = imgs.to(device=device)
        mask_gt = mask_gt.to(device=device)


        # save_path = os.path.join(save_dir, idx)
        # mkdir(save_path)

        with torch.no_grad():
            x1, x2, x3, x4, x5 = encoder(imgs)
            mask_pred = decoder(x1, x2, x3, x4, x5)

        loss += celoss(mask_pred, mask_gt)

        pred = torch.argmax(mask_pred, 1)

        miou = compute_mean_iou(pred.squeeze().cpu().numpy().flatten().astype(np.uint8), mask_gt.squeeze().cpu().numpy().flatten().astype(np.uint8))
        miou_list.append(miou)

        if last:
            pred_img = pred.squeeze().float().cpu().numpy()*255
            cv2.imwrite(os.path.join(save_dir, f'{idx}_pred.png'), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        
        # if current_step % 10000 == 0:
        # border = torch.ones((mask_gt.squeeze().shape[0], 5)).type(torch.uint8)
        # single_tensor = torch.hstack((pred.squeeze().cpu(), border, mask_gt.squeeze().cpu())) * 255
        # # tensor_list.append(single_tensor.type(torch.long))
        # img_comb = transforms.ToPILImage()(single_tensor.type(torch.uint8)).convert("L")
        # img_comb.save(os.path.join(save_path, f'{idx}__{current_step}.png'))       

    # if current_step % 6000 == 0:
    #     wandb.log({'Output': [wandb.Image(img.numpy()) for img in tensor_list]}, step=current_step)
    encoder.train()
    decoder.train()
    return loss / len(loader) , np.mean(miou_list) 
