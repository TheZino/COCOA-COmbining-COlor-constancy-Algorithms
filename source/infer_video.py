import argparse
import csv
import os
import time
import math
import numpy as np

from utils.ColorMetrics import RecoveryLoss
import torch
from torch.utils.data import DataLoader

from data.dataset_video import TestsetFromFolder, VideoSampler, collate_pad_frames
from models.combonn import ComboNN_video_pre as ComboNN
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
    "--in_dir", type=str, default="", help="directory containing input images"
)
parser.add_argument(
    "--model", type=str, default="", help="Pre-trained generator model path"
)
parser.add_argument(
    "--inest", type=int, default=18, help="number of input feature of the model"
)
parser.add_argument(
    "--hlnum", type=int, default=3, help="number of hidden feature of the model"
)
parser.add_argument(
    "--hlweights", nargs="*", default=["256", "128"], help="weights of each level"
)
parser.add_argument(
    "--lstm_nch",
    type=int,
    default=64,
    help="number of hidden feature of the lstm module",
)

parser.add_argument(
    "--threads", type=int, default=4, help="number of threads for data loader to use"
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


model = ComboNN(
    in_nch=opt.inest,
    hlnum=opt.hlnum,
    hlweights=opt.hlweights,
    lstm_nch=opt.lstm_nch,
    out_nch=3,
)

print(model)

### Models loading
print("\n===> Loading Model")

ckpt = torch.load(model_w, map_location="cpu")

model.load_state_dict(ckpt["model_state_dict"])

### Criterions
Loss_crit = RecoveryLoss()

############################ Setting cuda ######################################

print("\n===> Setting GPU")

if cuda_check:
    model.to(device)

############################### TEST ###########################################

tset = TestsetFromFolder(in_dir)
test_sampler = VideoSampler(tset, shuffle=False)
data_loader = DataLoader(
    dataset=tset,
    num_workers=opt.threads,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_pad_frames,
    sampler=test_sampler,
)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

log_file = open(opt.save_file, "w")
wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)

# error = 0
errors = []

model.eval()

with torch.no_grad():
    i = 0

    print("Processing " + str(i + 1) + "/" + str(len(data_loader)), end="\r")
    i += 1

    for i, batch in enumerate(data_loader, 1):

        inputt = batch[0]
        target = batch[1]
        mask = batch[2]
        X_lengths = batch[3]
        name = batch[4]

        if cuda_check:
            inputt = inputt.to(device)
            target = target.to(device)
            mask = mask.to(device)

        sttime = time.time()
        out = model(inputt, mask, X_lengths)
        end = time.time() - sttime

        ### Metrics Evaluation
        errors.append(Loss_crit(out, target).item())

        printProgressBar(i, len(data_loader), prefix="Test:", suffix="", length=50)

        results = [
            name,
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


print("Time per image: {:.8f}".format(end))
print(
    "\n\tMean angular error: {:.8f}\n\tMedian angular error: {:.8f}".format(
        np.mean(errors) * 180 / math.pi, np.median(errors) * 180 / math.pi
    )
)

print("\n\nFinished")
