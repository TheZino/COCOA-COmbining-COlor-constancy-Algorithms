import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.ColorMetrics import RecoveryLoss
from data.dataset_si import DatasetFromFolder, ValsetFromFolder
from models.combonn import ComboNN_single as ComboNN
from utils.print_utils import printProgressBar
from utils.general_utils import adjust_learning_rate, save_model, save_optim
from utils.weight_initializers import init_weights

##################################################################################################################################

parser = argparse.ArgumentParser(description="Training")

parser.add_argument(
    "--epochs", type=int, default=2000, help="number of epochs to train for"
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

parser.add_argument("--lr", type=float, default=1e-3, help="Starting Learning Rate.")
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
parser.add_argument("--infeat", type=int, default=24, help="number of feature in input")
parser.add_argument(
    "--hfeat", type=int, default=16, help="number of hidden feature of the model"
)
parser.add_argument(
    "--hlnum", type=int, default=2, help="number of hidden feature of the model"
)
parser.add_argument(
    "--hlweights", nargs="*", help="weights of each level", required=True
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
torch.manual_seed(123123)
if cuda_check:
    torch.cuda.manual_seed(123123)

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
training_data_loader = DataLoader(
    dataset=train_set,
    num_workers=opt.threads,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)

val_set = ValsetFromFolder(validation_image_path)
validation_data_loader = DataLoader(
    dataset=val_set,
    num_workers=opt.threads,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)

step_n = int(len(train_set) / batch_size)

print(
    "\nDataset: {}\nNumber of images: {}\nNumber of iteration per epoch: {}".format(
        input_image_path, len(train_set), step_n
    )
)

############################ Network Models ####################################

print("\n===> Building models")

model = ComboNN(in_nch=opt.infeat, hlnum=opt.hlnum, hlweights=opt.hlweights, out_nch=3)

init_weights(model, "normal")

### Models loading
if model_chkp != "":
    print("\n===> Loading Model")
    model.load_state_dict(torch.load(pretrained_G))

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

        if cuda_check:
            inputt = inputt.to(device)
            target = target.to(device)

        out = model(inputt)

        ### Loss evaluation

        # MSE Loss
        MSEG = Loss_crit(out, target)

        Loss = MSEG

        Loss.backward()
        optim.step()

        mse_loss_mean += MSEG.item()

        if i % 50 == 0:
            print(
                "[{}][{}/{}] Loss MSE: {:.8f}".format(
                    epoch, i, len(training_data_loader), MSEG.item()
                )
            )

    mse_loss_mean = mse_loss_mean / len(training_data_loader)
    info_str = [epoch, mse_loss_mean]


def validation(epoch):

    print("\n\n==> VALIDATION")

    global info_str
    global best_valid
    global save

    model.eval()
    err_mean = 0

    printProgressBar(
        0, len(validation_data_loader), prefix="Validation:", suffix="", length=50
    )

    for i, batch in enumerate(validation_data_loader, 1):

        inputt = batch[0]
        target = batch[1]
        name = batch[2]

        if cuda_check:
            inputt = inputt.to(device)
            target = target.to(device)

        out = model(inputt)

        err = Loss_crit(out, target)
        err_mean += err.item()

        ### Metrics Evaluation

        printProgressBar(
            i, len(validation_data_loader), prefix="Validation:", suffix="", length=50
        )

    err_mean = err_mean / len(validation_data_loader)

    if err_mean < best_valid:
        best_valid = err_mean
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "mean_error": err_mean,
        }
        torch.save(state, check_path + "/netComboNN_epochbest.pth")

    print("Validation mean recovery error: {}".format(err_mean))
    info_str += [err_mean]


### Main

for epoch in range(0, n_epochs):

    if str(epoch) in opt.lrdwn:
        print("\nDecreasing LR\n")
        adjust_learning_rate(optim, sf=0.1)

    train(epoch)

    with torch.no_grad():
        save = False
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
