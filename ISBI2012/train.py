import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
import imageio
import os
from tifffile import *
import subprocess
import argparse
import ttach as tta
from collections import OrderedDict


parser = argparse.ArgumentParser(description='PyTorch ISBI2012 Example')
parser.add_argument('--out_dim', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4, metavar='N',help='input batch size for training (default: 16)')
parser.add_argument('--val_size', type=int, default=4, metavar='N',help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay', type=str, default='exp',choices=['multistep','exp'])
parser.add_argument('--dropout_rate', type=float, default=0.0, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--pooling_mode', type=str, default='avg',choices=['avg','max'])
parser.add_argument('--num_fc', type=int, default=3, metavar='N',help='number of fc layers')
parser.add_argument('--training_mode', type=str, default='train',choices=['train','train+val'])
parser.add_argument('--pad_size', type=int, default=48)
parser.add_argument('--rotate', type=str, default='all',choices=['all','90'])
parser.add_argument('--elastic_level', type=int, default=3)
parser.add_argument('--noise_std', type=float, default=0.1, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--final_act', type=str, default="sigmoid", choices=["sigmoid",'tanh'])
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--loss', type=str, default="CE", choices=["CE",'l1'])
parser.add_argument('--predict_val_file', type=str, default='predict_val_label.tif')
parser.add_argument('--predict_test_file', type=str, default='test_label.tif')
parser.add_argument('--model', type=str, default="V", choices=["SO3","A5","S4","S4_V","A4","V","V3","V2"])
parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd", "adamW"])
parser.add_argument('--ckpt_path', type=str, default='./result/checkpoint')
parser.add_argument('--ckpt_name', type=str, default='baseline.ckpt')
parser.add_argument('--tta', type=bool, default = True)

def lr_scheduler_exponential(optimizer, epoch, init_lr, epoch_start, base_factor, verbose=False, printfct=print):
    """
    Decay initial learning rate exponentially starting after epoch_start epochs
    The learning rate is multiplied with base_factor every lr_decay_epoch epochs
    :param optimizer: the optimizer inheriting from torch.optim.Optimizer
    :param epoch: the current epoch
    :param init_lr: initial learning rate before decaying
    :param epoch_start: epoch after which the learning rate is decayed exponentially. Constant lr before epoch_start
    :param base_factor: factor by which the learning rate is decayed in each epoch after epoch_start
    :param verbose: print current learning rate
    """
    if epoch <= epoch_start:
        lr = init_lr * .1
    else:
        lr = init_lr * (base_factor**(epoch - epoch_start))
    if verbose:
        printfct('learning rate = {:6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr

def lr_scheduler_multistep(optimizer, epoch, init_lr, epoch_start, verbose=False):
    """
    Decay initial learning rate exponentially starting after epoch_start epochs
    The learning rate is multiplied with base_factor every lr_decay_epoch epochs
    :param optimizer: the optimizer inheriting from torch.optim.Optimizer
    :param epoch: the current epoch
    :param init_lr: initial learning rate before decaying
    :param epoch_start: epoch after which the learning rate is decayed exponentially. Constant lr before epoch_start
    :param base_factor: factor by which the learning rate is decayed in each epoch after epoch_start
    :param verbose: print current learning rate
    """
    if epoch <= epoch_start:
        lr = init_lr * .1
    elif epoch <= 80:
        lr = init_lr
    elif epoch <= 140:
        lr = init_lr * 0.1
    elif epoch <= 200:
        lr = init_lr * 0.01
    else:
        lr = init_lr * 0.001
    if verbose:
        print('learning rate = {:6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr

# Network
def train(net,args,train_loader,val_in_train_loader,val_loader,test_loader):
    if args.loss == 'CE':
        loss_fn = nn.BCELoss().cuda()
    elif args.loss == "l1":
        loss_fn = nn.SmoothL1Loss().cuda()

    # Optimizer
    optim = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    num_epochs = args.epochs
    net.train()  # Train Mode
    train_loss_arr = list()
    v_rands = []
    v_infos = []
    best_v_rand = 0
    best_v_info = 0
    for epoch in range(args.epochs):
        train_batch_num = len(train_loader)
        for batch_idx, data in enumerate(train_loader):
            net.train()
            img = data['img'].cuda()
            label = data['label'].cuda()
            output = net(img)
            if args.lr_decay == 'exp':
                optim, _ = lr_scheduler_exponential(optim, epoch, init_lr=args.lr, epoch_start=5, base_factor=.99, verbose=False)
            elif args.lr_decay == 'multistep':
                optim, _ = lr_scheduler_multistep(optim, epoch, init_lr=args.lr, epoch_start=5, verbose=False)

            optim.zero_grad()

            loss = loss_fn(output,label)
            loss.backward()

            optim.step()

            # Calc Loss Function
            train_loss_arr.append(loss.item())
            print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch+1, args.epochs, batch_idx, train_batch_num, train_loss_arr[-1]))

        if epoch % 5 == 0:
            v_rand, v_info = eval(args,net,val_loader,loss_fn,use_tta=True,mode='val')
            v_rands.append(v_rand)
            v_infos.append(v_info)
            if v_rand > best_v_rand:
                print('At epoch %d, get util now best v_ran: %s, v_inf: %s' % (epoch+1, v_rand, v_info))
                best_v_rand = v_rand
                save_dict = net.state_dict()
                print('Saving until now best checkpoint...')
                torch.save(save_dict, os.path.join(args.ckpt_path, 'best_'+args.ckpt_name))
                test(net,args,test_loader)

            if v_info > best_v_info:
                best_v_info = v_info
                
            id1 = v_infos.index(max(v_infos))
            id2 = v_rands.index(max(v_rands))
#     print('=== finally: ===')
            print('epoch %d V_Rand: %s  V_info: %s' %(id1*5, v_rands[id1], v_infos[id1]))
            print('epoch %d V_Rand: %s  V_info: %s' %(id2*5, v_rands[id2], v_infos[id2]))


def eval(args,net,val_loader,loss_fn,use_tta,mode):
    with torch.no_grad():
        if use_tta:
            net = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode='mean')
        
        net.eval()  # Evaluation Mode
        val_loss_arr = list()
        val_label = np.zeros((5,512,512))
        tmp = []
        for batch_idx, data in enumerate(val_loader):
            # Forward Propagation
            img = data['img'].cuda()
            label = data['label'].cuda()
            output = net(img)
            
            # Calc Loss Function
            loss = loss_fn(output, label)
            val_loss_arr.append(loss.item())
            
            print_form = '[Validation] | Loss: {:.4f}'
            print(print_form.format(val_loss_arr[-1]))  
            
            output = output[:, :, :, args.pad_size:128+args.pad_size, args.pad_size:128+args.pad_size]
            output = output.cpu().numpy()
            tmp.append(output)
            if args.val_size * len(tmp) == 16:                
                output = np.concatenate(tmp,axis=0)
                val_label[batch_idx*args.val_size//16*5:(batch_idx*args.val_size//16+1)*5,:,:] = output.reshape(4,4,5,128,128).transpose(2,0,3,1,4).reshape(5,512,512)
                tmp = []
    current_address = os.path.dirname(os.path.abspath(__file__))
    if mode == 'val_in_train':
        original_label = current_address+'/result/val_in_train_label.tif'
    elif mode == 'val':
        original_label = current_address+'/result/val_label.tif'

    proposal_label = current_address+'/result/' + args.predict_val_file
    imwrite(proposal_label,val_label.astype(np.float32))

    Fiji_path = current_address + '/result/Fiji.app/ImageJ'  # ImageJ-linux64'
    script_path = current_address + '/result/isbi12_eval_script.bsh'
    tmp_file = current_address + '/result/tmp.txt'
    return_info = subprocess.Popen([Fiji_path, script_path, original_label, proposal_label, tmp_file],
                                  shell=False, stdout=subprocess.PIPE)
    print('===============')
    v_rand, v_info = 0, 0
    while return_info.poll() is None:
        line = return_info.stdout.readline()
        line = line.strip().decode('utf-8')  # bytes to str
        if line:
            print(line)
            if 'Best' in line:
                if line[10:14] == 'Rand':
                    # v_rand = 0 if 'E-' in line else float(line[17:24])
                    if 'E-' in line:
                        v_rand = 0
                    else:
                        try:
                            v_rand = float(line[17:24])
                        except ValueError as ve:
                            v_rand = 0
                            print('===> ValueError: may due to: could not convert string to float.')
                elif line[10:14] == 'info':
                    # v_info = 0 if 'E-' in line else float(line[17:24])
                    if 'E-' in line:
                        v_info = 0
                    else:
                        try:
                            v_info = float(line[17:24])
                        except ValueError as ve:
                            v_info = 0
                            print('===> ValueError: may due to: could not convert string to float.')
    print('===============')
    return v_rand, v_info

def test(net, args, test_loader):
    with torch.no_grad():
        save_dict = torch.load(os.path.join(args.ckpt_path, 'best_' + args.ckpt_name))
        net.load_state_dict(save_dict)

        if args.tta:
            net = tta.SegmentationTTAWrapper(net, tta.aliases.d4_transform(), merge_mode='mean')

        net.eval()  # Evaluation Mode
        test_label = np.zeros((30, 512, 512))
        tmp = []
        for batch_idx, data in enumerate(test_loader):
            # Forward Propagation
            img = data['img'].cuda()
            output = net(img).cpu().numpy()
            output = output[:, :, :, args.pad_size:128+args.pad_size, args.pad_size:128+args.pad_size]
            tmp.append(output)
            if args.val_size * len(tmp) == 16:                
                output = np.concatenate(tmp,axis=0)
                test_label[batch_idx*args.val_size//16*5:(batch_idx*args.val_size//16+1)*5, :, :] = output.reshape(4,4,5,128,128).transpose(2,0,3,1,4).reshape(5,512,512)
                tmp = []

        current_address = os.path.dirname(os.path.abspath(__file__))
        proposal_label = current_address + '/result/' + args.predict_test_file
        imwrite(proposal_label, test_label.astype(np.float32))

def main():
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    device = torch.device("cuda")
    print("model:",args.model)
    elif args.model == "S4":
        from model.S4_Net import Model
    elif args.model == "V":
        from model.V_Net import Model
    else:
        raise AssertionError("Wrong model!")

    network = Model(args.out_dim,args.final_act,args.dropout_rate,args.pooling_mode,args.num_fc)
    network.cuda()
    network = torch.nn.DataParallel(network)



    num_para = torch.cat([p.view(-1) for p in network.parameters()]).size()

    print("number of parameters:", num_para)
    # Set Dataset

    train_transform = transforms.Compose([
        ReflectPadding(args.pad_size),
        Rotate(limit=180),
        RandomElasticDeform(spatial_crop_size=128+args.pad_size*2,points=args.elastic_level),
        RandomFlip(),
        ToTensor(),
        GaussianNoise(args.noise_std),
    ])
    
    val_transform = transforms.Compose([
        ReflectPadding(args.pad_size),
        ToTensor(),
    ])
    val_in_train_transform = val_transform
    test_transform = val_transform

    train_dataset = Dataset(mode=args.training_mode,transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_in_train_dataset = Dataset(mode="val_in_train",transform=val_in_train_transform)
    val_in_train_loader = DataLoader(val_in_train_dataset, batch_size=args.batch_size, shuffle=False)    
    val_dataset = Dataset(mode="val",transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.val_size, shuffle=False)
    test_dataset = Dataset(mode="test",transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.val_size, shuffle=False)
    
    train(network,args,train_loader,val_in_train_loader,val_loader,test_loader)


if __name__ == "__main__":
    main()
    
    


