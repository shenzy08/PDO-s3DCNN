# pylint: disable=E1101,R,C
import os
import numpy as np
import shutil
import requests
import zipfile
from subprocess import check_output
import torch
import types
import importlib.machinery
import sys

sys.path.append("..")

from shapes import Shrec17, CacheNPY, Obj2Voxel, EqSampler

from util import low_pass_filter


class KeepName:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, file_name):
        return file_name, self.transform(file_name)


def main(log_dir, augmentation, dataset, batch_size, num_workers, filename):
    print(check_output(["node", "--version"]).decode("utf-8"))

    torch.backends.cudnn.benchmark = True
    cache = CacheNPY("v64d", transform=Obj2Voxel(64, double=True, rotate=False), repeat=augmentation, pick_randomly=False)
    def transform(x):
        xs = cache(x)
        xs = [torch.from_numpy(x.astype(np.float32)).unsqueeze(0) / 8 for x in xs]
        xs = torch.stack(xs)
        return xs

    transform = KeepName(transform)

    test_set = Shrec17("shrec17_data", dataset, perturbed=True, download=True, transform=transform)

    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    
    model = mod.Model()
    model = model.cuda()
    model=torch.nn.DataParallel(model)

    state = torch.load(os.path.join(log_dir, filename))
    state = { key.replace('conv.kernel_', 'kernel.kernel_').replace('conv.weight', 'kernel.weight') : value for key, value in state.items() }
    model.load_state_dict(state)

    resdir = os.path.join(log_dir, dataset + "_perturbed")
    if os.path.isdir(resdir):
        shutil.rmtree(resdir)
    os.mkdir(resdir)

    predictions = []
    ids = []

    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    print("len of loader:", len(loader))
    for batch_idx, data in enumerate(loader):
        model.eval()

        if dataset != "test":
            data = data[0]

        file_names, data = data
        batch_size, rep = data.size()[:2]
        data = data.view(-1, *data.size()[2:])

        data = data.cuda()
        with torch.no_grad():
            pred = model(data)
        pred = pred.view(batch_size, rep, -1)
        pred = pred.sum(1)

        predictions.append(pred.detach().cpu().numpy())
        ids.extend([x.split("/")[-1].split(".")[0] for x in file_names])

        print("[{}/{}]      ".format(batch_idx, len(loader)))

    predictions = np.concatenate(predictions)

    predictions_class = np.argmax(predictions, axis=1)

    print("write files...")
    for i in range(len(ids)):
        if i % 100 == 0:
            print("{}/{}    ".format(i, len(ids)), end="\r")
        idfile = os.path.join(resdir, ids[i])

        retrieved = [(predictions[j, predictions_class[j]], ids[j]) for j in range(len(ids)) if predictions_class[j] == predictions_class[i]]
        retrieved = sorted(retrieved, reverse=True)
        retrieved = [i for _, i in retrieved]

        with open(idfile, "w") as f:
            f.write("\n".join(retrieved))

    print("nodejs script...")
    output = check_output(["node", "evaluate.js", os.path.join("..", log_dir) + "/"], cwd="evaluator").decode("utf-8")
    print(output)
    shutil.copy2(os.path.join("evaluator", log_dir + ".summary.csv"), os.path.join(log_dir, "summary.csv"))

    _, p, r, f, mAP, ndcg, _, _ = next(line for line in output.splitlines() if 'microALL' in line).split(',')
    micro = {
        "P": float(p), "R": float(r), "F1": float(f), "mAP": float(mAP), "NDCG": float(ndcg)
    }
    _, p, r, f, mAP, ndcg, _, _ = next(line for line in output.splitlines() if 'macroALL' in line).split(',')
    macro = {
        "P": float(p), "R": float(r), "F1": float(f), "mAP": float(mAP), "NDCG": float(ndcg)
    }
    return micro, macro


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--filename", type=str, default="state.pkl")

    args = parser.parse_args()

    main(**args.__dict__)


