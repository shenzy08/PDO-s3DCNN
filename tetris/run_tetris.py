#!/bin/bash
# pylint: disable=C,R,E1101,E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import sys
sys.path.append("..")
sys.path.append("...")
from conf import settings
from scipy.ndimage import zoom, affine_transform
import lr_schedulers
from lr_schedulers import lr_scheduler_exponential
import argparse
import random
import time

parser = argparse.ArgumentParser(description='PyTorch Tetris Example')
parser.add_argument('--seed', type=int, default=0, metavar='S',help='random seed (default: 1)')
parser.add_argument('--model', type=str, default="S4", choices=["S4", "S4_A4", "S4_V", "SO3"])
parser.add_argument('--discretized_mode', type=str, default="FD", choices=["FD","Gaussian"])
parser.add_argument('--drop_rate', type=float, default=0.0)

def rotate_scalar(x, rot):
    # this function works in the convention field[x, y, z]
    invrot = np.linalg.inv(rot)
    center = (np.array(x.shape) - 1) / 2
    return affine_transform(x, matrix=invrot, offset=center - np.dot(invrot, center))

def rot_z(gamma):
    '''
    Rotation around Z axis
    '''
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=torch.get_default_dtype())
    return gamma.new_tensor([
        [gamma.cos(), -gamma.sin(), 0],
        [gamma.sin(), gamma.cos(), 0],
        [0, 0, 1]
    ])


def rot_y(beta):
    '''
    Rotation around Y axis
    '''
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=torch.get_default_dtype())
    return beta.new_tensor([
        [beta.cos(), 0, beta.sin()],
        [0, 1, 0],
        [-beta.sin(), 0, beta.cos()]
    ])


def rot(alpha, beta, gamma):
    '''
    ZYZ Eurler angles rotation
    '''
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def get_volumes(size=20, pad=10, rotate=False, rotate90=False):
    assert size >= 4
    tetris_tensorfields = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]

    labels = np.arange(len(tetris_tensorfields))

    tetris_vox = []
    for shape in tetris_tensorfields:
        volume = np.zeros((4, 4, 4))
        for xi_coords in shape:
            volume[xi_coords] = 1

        volume = zoom(volume, size / 4, order=0)
        volume = np.pad(volume, pad, 'constant')

        if rotate:
            a, c = np.random.rand(2) * 2 * np.pi
            b = np.arccos(np.random.rand() * 2 - 1)
            volume = rotate_scalar(volume, rot(a, b, c))
        if rotate90:
            volume = rot_volume_90(volume)

        tetris_vox.append(volume[np.newaxis, ...])

    tetris_vox = np.stack(tetris_vox).astype(np.float32)

    return tetris_vox, labels

def rot_volume_90(vol):
    k1, k2, k3 = np.random.randint(4, size=3)
    vol = np.rot90(vol, k=k1, axes=(0, 1))  # z
    vol = np.rot90(vol, k=k2, axes=(0, 2))  # y
    vol = np.rot90(vol, k=k3, axes=(0, 1))  # z
    return vol

def train(network, dataset, N_epochs):
    network.train()

    volumes, labels = dataset
    volumes = torch.tensor(volumes).cuda()
    labels = torch.tensor(labels).cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=2e-2)
    record_loss = np.zeros(N_epochs)
    time_before = time.perf_counter()

    for epoch in range(N_epochs):
        predictions = network(volumes)

        optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, init_lr=1e-2, epoch_start=50, base_factor=.98, verbose=False)
#         for 

        optimizer.zero_grad()
        loss = F.cross_entropy(predictions, labels)
        loss.backward()
        optimizer.step()
        record_loss[epoch] = loss

        argmax = predictions.argmax(1)
        acc = (argmax.squeeze() == labels).float().mean().item()
        print('epoch {}: loss={}, acc={:.3f}'.format(epoch, loss, acc), end=" \r")
    
    np.save("loss.npy",record_loss)
    time_after = time.perf_counter()
    print("total time: {}, time per iteration: {}".format(time_after-time_before,(time_after-time_before)/N_epochs))
        
def test(network, dataset):
    network.eval()

    volumes, labels = dataset
    volumes = torch.tensor(volumes).cuda()
    labels = torch.tensor(labels).cuda()

    predictions = network(volumes)
    argmax = predictions.argmax(1)
    acc = (argmax.squeeze() == labels).float().mean().item()

    print('test acc={:.3f}'.format(acc), end=" \r")
    return acc


def experiment(network):
    N_epochs = 200
    N_test = 100
    trainset = get_volumes(rotate=False)  # train with randomly rotated pieces but only once

    train(network, trainset, N_epochs=N_epochs)
    
    with torch.no_grad():     
        
        test_accs = []
        for _ in range(N_test):
            testset = get_volumes(rotate=True)
            acc = test(network, testset)
            test_accs.append(acc)
    return np.mean(test_accs)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    setup_seed(args.seed)
    device = torch.device("cuda")

    print("model:",args.model)
    if args.model == "S4":
        from model.S4_Net import S4_Net
        network = S4_Net()
    elif args.model == "S4_A4":
        from model.S4_A4_Net import S4_A4_Net
        network = S4_A4_Net()
    elif args.model == "S4_V":
        from model.S4_V_Net import S4_V_Net
        network = S4_V_Net()
    elif args.model == "SO3":
        from model.SO3_Net import SO3_Net
        network = SO3_Net(args.discretized_mode,args.drop_rate)
    else:
        raise AssertionError("Wrong model!")

    network.to(device)
    num_para = torch.cat([p.view(-1) for p in network.parameters()]).size()
    print("number of parameters:", num_para)
    acc = experiment(network)
    print('\n')
    print("acc={}".format(acc))


if __name__ == "__main__":
    main()

