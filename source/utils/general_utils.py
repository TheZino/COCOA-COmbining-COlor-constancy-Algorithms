import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(SEED):
    # np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"]
    )


def save_model(net, path, epoch, name):
    torch.save(net, "{}/net{}_epoch{}.pth".format(path, name, str(epoch)))


def save_optim(optim, path, epoch, name):
    torch.save(optim, "{}/optim{}_epoch{}.pth".format(path, name, str(epoch)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, sf=0.1):
    """Sets the learning rate to the optimizer LR decayed by 10 """
    lr = optimizer.param_groups[0]["lr"]
    lr = lr * (0.1)
    print("\n \033[92m|NEW LR:\033[0m " + str(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            matplotlib.lines.Line2D([0], [0], color="c", lw=4),
            matplotlib.lines.Line2D([0], [0], color="b", lw=4),
            matplotlib.lines.Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()
