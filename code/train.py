#
import copy
import argparse

#
import wandb
import numpy as np
#
import torch
import torch.optim as optim

#
from models import Encoder, Decoder
from utils.loss import celoss
from utils.util import create_dataloader, set_random_seed, mkdir
from val import eval_net

def train(opt):
    #args
    ft, enc_weight_path, dec_weight_path, lr, data, ckpt_dir, val_dir, data_path, total_iters, device, print_frq, save_frq, eval_frq, test_dir = opt.finetune, opt.enc_weight, opt.dec_weight, \
        opt.lr, opt.data, opt.ckpt_dir, opt.val_dir, opt.data_path, opt.total_iters, opt.device, opt.print_frq, opt.save_frq, opt.eval_frq, opt.test_dir 
    
    #args wandb
    project, entity, run_name = opt.project, opt.entity, opt.run_name

    #random seed
    set_random_seed(8848)

    #write result to a file
    results = open('results.txt', 'w')
    print("Writing to a file")
    results.write('Iter\t MEAN\n')

    max_miou = 0
    best_ite = 0
    ite_num = 0

    #set up experiment directories
    mkdir(ckpt_dir)
    mkdir(val_dir)   
    mkdir(test_dir)

    #setup wandb
    wandb.init(project=project, entity=entity, name=run_name)
    # wandb.watch([net], log='all')

    #dataloader    
    print(f'------Training on {data} data for {total_iters} iterations ----llr: {lr}--ft: {ft}----')
    train_dataloader, val_dataloader = create_dataloader(data_path)

    #define model
    encoder = Encoder(3, 2, bilinear=False)
    decoder = Decoder(3, 2, bilinear=False)
    best_encoder = None
    best_decoder = None

    if torch.cuda.is_available():
        print(ft)
        if ft:
            encoder.load_state_dict(torch.load(enc_weight_path))
            decoder.load_state_dict(torch.load(dec_weight_path))
        encoder.to(device)
        encoder.train()
        decoder.to(device)
        decoder.train()

        if ft:
            for param in decoder.parameters():
                param.requires_grad = False

    #optimizer and scheduler

    #enc
    opt_encoder  = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    # sch_encoder = optim.lr_scheduler.ReduceLROnPlateau(opt_encoder, 'min', patience=10, factor=0.5)
    sch_encoder = optim.lr_scheduler.MultiStepLR(opt_encoder, [5000, 10000, 15000], 0.5)

    #dec
    if not ft:
        opt_decoder  = optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
        # sch_decoder = optim.lr_scheduler.ReduceLROnPlateau(opt_decoder, 'min', patience=10, factor=0.5) 
        sch_decoder = optim.lr_scheduler.MultiStepLR(opt_decoder, [5000, 10000,15000], 0.5) 
    

    while True:
        
        for i, data in enumerate(train_dataloader):
            ite_num = ite_num + 1
            inputs, labels = data['image'].to(device), data['mask'].to(device)
            
            if not ft:
                opt_decoder.zero_grad()
            opt_encoder.zero_grad()

            x1, x2, x3, x4, x5 = encoder(inputs)
            mask_pred = decoder(x1, x2, x3, x4, x5)

            loss = celoss(mask_pred, labels)

            loss.backward()
            if not ft:
                opt_decoder.step()  
                sch_decoder.step()          
            opt_encoder.step()
            sch_encoder.step() 
            
            if ite_num % print_frq == 0:
                print(f'Iter: {ite_num}\t Loss: {loss.item()}')

            wandb.log({"Train/loss":loss.item(), "lr": opt_decoder.param_groups[0]['lr']}, step=ite_num)
    
            if ite_num % save_frq == 0:
                print('saving_model')
                torch.save(encoder.state_dict(), f'{ckpt_dir}/encoder_{ite_num}.pth')
                torch.save(decoder.state_dict(), f'{ckpt_dir}/decoder_{ite_num}.pth')


            if ite_num % eval_frq == 0:
                print('Validating')            
                ce_loss_val, fold_iou = eval_net(encoder, decoder, val_dataloader, device, ite_num, wandb, val_dir)
                if ft:
                    for param in decoder.parameters():
                        param.requires_grad = False
                
                # if not ft:
                #     sch_decoder.step(ce_loss_val)
                
                # sch_encoder.step(ce_loss_val)

                if fold_iou.item() > max_miou:
                    max_miou = fold_iou.item()
                    
                    best_ite = ite_num
                    best_encoder = copy.deepcopy(encoder)
                    best_decoder = copy.deepcopy(decoder)


                wandb.log({"Valid/loss": ce_loss_val, 
                            "Metric/mIOU": fold_iou.item()}, 
                            step=ite_num)

                print(f'Ite: {ite_num}\t Val_loss: {ce_loss_val}\t MeanIOU: {fold_iou.item()}')


            if total_iters <= ite_num:
                torch.save(best_encoder.state_dict(), f'{ckpt_dir}/bestencoder_{best_ite}.pth')
                torch.save(best_decoder.state_dict(), f'{ckpt_dir}/bestdecoder_{best_ite}.pth')
                np.array([best_ite, max_miou]).tofile(results, sep="\t")
                results.write('\n')
                results.close()        
                break
  
        if total_iters <= ite_num:
            ce_loss_val, fold_iou = eval_net(best_encoder, best_decoder, val_dataloader, device, ite_num, wandb, test_dir, last=True)
            print(f'IoU: {fold_iou}')
            break

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='runs/checkpoints', type=str,  help='dir to save checkpoints')
    parser.add_argument('--val_dir', default='runs/val_results', type=str, help='dir to save validation results')
    parser.add_argument('--test_dir', default='runs/preds', type=str, help='dir to save best predictions results')
    parser.add_argument('--data_path', default='/raid/binod/projects/multicenter/ETIS/data/split', type=str, help='Data path')
    parser.add_argument('--data', default='', type=str, help='Data to train')
    parser.add_argument('--total_iters', default=100001, type=int, help='No of iterations to train for')
    parser.add_argument('--device', default=0, type=int, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--print_frq', default=1000, type=int, help='Print frequency')
    parser.add_argument('--save_frq', default=25000, type=int, help='Save frequency')
    parser.add_argument('--eval_frq', default=1000, type=int, help='Save frequency')
    parser.add_argument('--lr', default=0.0002, type=float, help='Save frequency')
    parser.add_argument('--finetune', default=False, type=bool, help='Finetune')
    parser.add_argument('--enc_weight', default='', type=str, help='Encoder Weight Path')
    parser.add_argument('--dec_weight', default='', type=str, help='Decoder Weight Path')

    #wandb
    parser.add_argument('--project', default='multicenter', type=str, help='wandb project name')
    parser.add_argument('--entity', default='naamii', type=str, help='wandb entity name')
    parser.add_argument('--run_name', default='runa', type=str, help='wandb run name')
    return parser.parse_args()

if __name__=='__main__':
    opt = parse_opt()
    print(opt)
    train(opt)

