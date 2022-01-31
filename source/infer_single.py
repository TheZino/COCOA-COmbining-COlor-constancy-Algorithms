import argparse
import csv
import os
import time
from os import listdir
from os.path import join

import skimage.io as io
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.dataset_si import TestsetFromFolder
from models.combonn import ComboNN_single as ComboNN
from utils.print_utils import printProgressBar


################################################################################
################################################################################
################################################################################


parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--save_file",
    type=str,
    default="./output/results.csv",
    help="directory where to save output data and checkpoints",
)

parser.add_argument(
    "--in_dir", type=str, default="./", help="directory containing input images"
)
parser.add_argument(
    "--model", type=str, default="", help="Pre-trained generator model path"
)
parser.add_argument(
    "--infeat", type=int, default=18, help="number of input feature of the model"
)
parser.add_argument(
    "--hfeat", type=int, default=16, help="number of hidden feature of the model"
)
parser.add_argument(
    "--hlnum", type=int, default=2, help="number of hidden feature of the model"
)
parser.add_argument(
    "--hlweights", nargs="*", help="weights of each level", required=True
)

parser.add_argument("--device", type=str, default="cpu", help="GPU or CPU device")


opt = parser.parse_args()
# print(opt)

##################### General Options ##########################################

in_dir = opt.in_dir
device = opt.device

save_dir = os.path.dirname(os.path.abspath(opt.save_file))

### Path to the pretrained model to load
model_w = opt.model


### GPU board availability check
cuda_check = torch.cuda.is_available()


############################ Network Models ####################################
print("\n===> Building models")


model = ComboNN(in_nch=opt.infeat, hlnum=opt.hlnum, hlweights=opt.hlweights, out_nch=3)

### Models loading
print("\n===> Loading Model")

ckpt = torch.load(model_w, map_location="cpu")

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

############################ Setting cuda ######################################

print("\n===> Setting GPU")

if cuda_check:
    model.to(device)

############################### TEST ###########################################

tset = TestsetFromFolder(in_dir)
data_loader = DataLoader(
    dataset=tset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True
)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

log_file = open(opt.save_file, "w")
wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)
mean_rmse = 0

with torch.no_grad():
    i = 0

    print("Processing " + str(i + 1) + "/" + str(len(data_loader)), end="\r")
    i += 1

    for i, batch in enumerate(data_loader, 1):

        inputt = batch[0]
        target = batch[1]
        name = batch[2]

        if cuda_check:
            inputt = inputt.to(device)
            target = target.to(device)

        sttime = time.time()
        out = model(inputt)
        end = time.time() - sttime
        ### Metrics Evaluation

        printProgressBar(
            i, len(data_loader), prefix="Validation:", suffix="", length=50
        )

        results = [
            name[0],
            target[0, 0].item(),
            target[0, 1].item(),
            target[0, 2].item(),
            out[0, 0].item(),
            out[0, 1].item(),
            out[0, 2].item(),
        ]

        wr.writerow(results)
        log_file.flush()
        results = []

print("time per image: {:.8f}".format(end))
print("Finished")
