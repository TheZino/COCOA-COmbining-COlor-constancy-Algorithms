import argparse
import csv
import os

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.ColorMetrics import RecoveryLoss
from data.dataset_video import (
    DatasetFromFolder,
    ValsetFromFolder,
    VideoSampler,
    collate_pad_frames,
)
from models.combonn import ComboNN_video as ComboNN
from utils.print_utils import printProgressBar
from utils.general_utils import adjust_learning_rate, seed_everything
from utils.weight_initializers import init_weights

##################################################################################################################################

parser = argparse.ArgumentParser(description="Training")

parser.add_argument(
    "--epochs", type=int, default=3000, help="number of epochs to train for"
)
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Device for training [cpu, cuda:0, cuda:1]",
)
parser.add_argument(
    "--threads", type=int, default=4, help="number of threads for data loader to use"
)

parser.add_argument("--lr", type=float, default=1e-4, help="Starting Learning Rate.")
parser.add_argument(
    "--lrdwn", nargs="*", help="Learning rate decreasing epochs", required=False
)

parser.add_argument(
    "--trainset_dir",
    type=str,
    default="./",
    help="directory containing training images",
)
parser.add_argument(
    "--validation_dir",
    type=str,
    default="./",
    help="directory containing validation images",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./",
    help="directory where to save output data and checkpoints",
)

parser.add_argument("--wd", action="store_true")
parser.add_argument("--inest", type=int, default=24, help="number of feature in input")
parser.add_argument(
    "--hlnum", type=int, default=3, help="number of hidden feature of the model"
)
parser.add_argument(
    "--lstm_nch",
    type=int,
    default=64,
    help="number of hidden feature of the lstm module",
)
parser.add_argument(
    "--hlweights",
    nargs="*",
    default=["256", "128"],
    help="weights of each level",
    required=True,
)

parser.add_argument("--model_chkp", type=str, default="", help="Model to load")
parser.add_argument("--optim_chkp", type=str, default="", help="Optimizer to load")

opt = parser.parse_args()


##################### General Options ##########################################
n_epochs = opt.epochs  # Total number of epochs for training
batch_size = opt.batch_size  # Image batch size
device = opt.device  # True: GPU training, False: CPU training


lr = opt.lr  # Starting Learning Rate
if opt.lrdwn is None:
    opt.lrdwn = []

input_image_path = opt.trainset_dir  # path to the input image directory
validation_image_path = opt.validation_dir  # path to the validation image directory

save_dir = opt.save_dir  # directory where to save checkpoints and outputs
check_path = save_dir + "/checkpoints"  # directory for checkpoint save

### Path to the pretrained model to load
model_chkp = opt.model_chkp
optim_chkp = opt.optim_chkp

### GPU board availability check
cuda_check = torch.cuda.is_available()

### Seed setting for random operations
seed_everything(123123)

##################### Check Folders ############################################

save_paths = [save_dir, check_path]

for pth in save_paths:
    if not os.path.exists(pth):
        os.makedirs(pth)

log_file = open(save_dir + "/log.csv", "w")
wr = csv.writer(log_file, quoting=csv.QUOTE_ALL)

##################### Options Print ############################################

print(
    "\n===> Options\n\nNumber of epochs: {}\nBatch size: {}\nDecreasing lr at: {}\nGPU device: {}\nSave directory: {}".format(
        n_epochs, batch_size, opt.lrdwn, device, save_dir
    )
)

##################### Batch Loader initialization ##############################

print("\n===> Loading Dataset")

train_set = DatasetFromFolder(input_image_path)
train_sampler = VideoSampler(train_set, shuffle=True)
training_data_loader = DataLoader(
    dataset=train_set,
    num_workers=opt.threads,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_pad_frames,
    sampler=train_sampler,
)

val_set = ValsetFromFolder(validation_image_path)
val_sampler = VideoSampler(val_set, shuffle=False)
validation_data_loader = DataLoader(
    dataset=val_set,
    num_workers=opt.threads,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_pad_frames,
    sampler=val_sampler,
)


print(
    "\nDataset: {}\nNumber of images: {}\nNumber of iteration per epoch: {}".format(
        input_image_path, len(train_set), len(training_data_loader)
    )
)

############################ Network Models ####################################

print("\n===> Building models")

model = ComboNN(
    in_nch=opt.inest,
    hlnum=opt.hlnum,
    hlweights=opt.hlweights,
    lstm_nch=opt.lstm_nch,
    out_nch=3,
)

init_weights(model, "normal")

### Models loading
if model_chkp != "":
    print("\n===> Loading Model")
    # model.load_mlp(model_chkp)
    model.load_state_dict(torch.load(model_chkp)["model_state_dict"])

########################### Setting Loss #######################################

### Criterions
Loss_crit = RecoveryLoss()


### Optimizers
if opt.wd:
    optim = optim.Adam(model.parameters(), lr=lr, weight_decay=lr / 200)
else:
    optim = optim.Adam(model.parameters(), lr=lr)

### Loading optimizers
if optim_chkp != "":
    print("\n===> Loading Optimizer")
    optim.load_state_dict(torch.load(optim_chkp))

############################ Setting cuda ######################################

print("\n===> Setting GPU")

if cuda_check:
    model.to(device)
    Loss_crit.to(device)


############################### TRAIN ##########################################
print("\n===> Training")

best_valid = 10000000
best_med_valid = 10000000
info_str = []
save = False


def train(epoch):
    global info_str
    model.train()
    print("\n")

    mse_loss_mean = 0

    for i, batch in enumerate(training_data_loader):

        model.zero_grad()

        # DATA
        inputt = batch[0]
        target = batch[1]
        masks = batch[2]
        X_lengths = batch[3]

        if cuda_check:
            inputt = inputt.to(device)
            target = target.to(device)
            masks = masks.to(device)

        out = model(inputt, masks, X_lengths)

        ### Loss evaluation

        # MSE Loss
        MSEG = Loss_crit(out, target)

        Loss = MSEG

        Loss.backward()
        optim.step()

        mse_loss_mean += MSEG.item() * 180 / math.pi

        if i % 50 == 0:
            print(
                "[{}][{}/{}] Loss MSE: {:.8f}".format(
                    epoch, i, len(training_data_loader), MSEG.item()
                ),
                end="\r",
            )

    mse_loss_mean = mse_loss_mean / len(training_data_loader)
    info_str = [epoch, mse_loss_mean]


def validation(epoch):

    global info_str
    global best_valid
    global best_med_valid

    model.eval()
    errors = []

    printProgressBar(
        0, len(validation_data_loader), prefix="Validation:", suffix="", length=50
    )
    with torch.no_grad():
        for i, batch in enumerate(validation_data_loader, 1):

            inputt = batch[0]
            target = batch[1]
            masks = batch[2]
            X_lengths = batch[3]

            if cuda_check:
                inputt = inputt.to(device)
                target = target.to(device)
                masks = masks.to(device)

            out = model(inputt, masks, X_lengths)

            err = Loss_crit(out, target)
            errors.append(err.item())

            ### Metrics Evaluation

            printProgressBar(
                i,
                len(validation_data_loader),
                prefix="Validation:",
                suffix="",
                length=50,
            )

    err_mean = np.mean(errors) * 180 / math.pi
    err_med = np.median(errors) * 180 / math.pi

    if err_mean < best_valid:
        best_valid = err_mean
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "mean_error": err_mean,
        }
        torch.save(state, check_path + "/netComboNN_epochbest.pth")

    if err_med < best_med_valid:
        best_med_valid = err_med
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "mean_error": err_mean,
        }
        torch.save(state, check_path + "/netComboNN_epochbest_med.pth")

    print("\t\t\t\t\t\t\t\t", end="\r")
    print(
        "Validation \n\tMean recovery error: \t{} \n\tMedian recovery error: \t{}".format(
            err_mean, err_med
        )
    )
    info_str += [err_mean, err_med]


### Main

for epoch in range(0, n_epochs):

    if str(epoch) in opt.lrdwn:
        print("\nDecreasing LR\n")
        adjust_learning_rate(optim, sf=0.1)

    train(epoch)

    validation(epoch)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "mean_error": info_str[-1],
    }
    torch.save(state, check_path + "/netComboNN_epochlast.pth")

    wr.writerow(info_str)
    log_file.flush()

    print("\nEpoch {} finished!".format(epoch))


log_file.close()
