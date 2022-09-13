import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with_latex_style = {
    'text.usetex': True,
    'font.family': 'stixgeneral',
    'mathtext.fontset': 'stix',
    "axes.labelsize": 28,
    "font.size": 30,
    "legend.fontsize": 24,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fancybox": False,
    "lines.linewidth": 1.0,
    "patch.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 150
    }

matplotlib.rcParams.update(with_latex_style)

d = np.load("finalNoSummarye11000.npy")
d_sn = np.load("finalSummary1Layer11000e300NodesCdim100.npy")

def plt_loss(data, name):
    cutoff = 100
    fig = plt.figure(figsize=(12,8))
    tr, te = data[cutoff:, 0], data[cutoff:, 1] # third one is learning rate
    x = np.arange(cutoff, data.shape[0])
    #plt.yscale("symlog")
    plt.title(f"The test and validation losses\n (from {cutoff} epochs)")
    plt.plot(x, te, label="Test loss")
    plt.plot(x, tr, label="Training loss")
    plt.axvline(np.argmin(te)+cutoff, 0, 1, color="red", linestyle="--")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(name)
    plt.show()
    
def plt_loss_SN(data, name):
    cutoff = 100
    fig = plt.figure(figsize=(12,8))
    tr, te = data[cutoff:, 0], data[cutoff:, 1] # third one is learning rate
    x = np.arange(cutoff, data.shape[0])
    plt.yscale("symlog")
    plt.title(f"The test and validation losses\n (from {cutoff} epochs)")
    plt.plot(x, te, label="Test loss")
    plt.plot(x, tr, label="Training loss")
    plt.axvline(np.argmin(te)+cutoff, 0, 1, color="red", linestyle="--")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(name)
    plt.show()

plt_loss(d, "losses")
plt_loss_SN(d_sn, "losses_SN")
