import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

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

def plt_bkg(ax):
    x = np.arange(0,3,0.01)
    bkg = scipy.stats.lognorm.pdf(x, loc=0.05, s=0.25)
    samples = np.random.lognormal(mean=0.05, sigma=0.25, size=100000)
    ax.hist(samples, bins=np.linspace(0,3,40), density=True)
    ax.plot(x, bkg, label="Lognormal $\mu=0.05, \sigma=0.25$")
    ax.axvline(samples.mean(), 0, 1, color="red", alpha=0.6, linestyle="--", label="mean: {:.2f}".format(samples.mean()))
    ax.legend()
    #ax.set_xlabel("x")

def plt_signal(ax):
    x = np.arange(0, 40, 0.01)
    y = scipy.stats.gamma.pdf(x, a=1.5, scale=7)
    samples = np.random.gamma(1.5, 7, size=1000000)
    ax.hist(samples, density=True, bins=np.linspace(0,40,40))
    ax.plot(x, y, label=r"$\Gamma(x; k=1.5, \theta=7)$")
    ax.axvline(samples.mean(), 0, 1, color="red", alpha=0.6, linestyle="--", label="mean: {:.2f}".format(samples.mean()))
    ax.legend()
    ax.set_xlabel("x")

def plt_lumi(ax):
    x = np.arange(0.9,1.1,0.001)
    bkg = scipy.stats.lognorm.pdf(x, loc=0.0, s=0.02)
    samples = np.random.lognormal(mean=0.0, sigma=0.02, size=100000)
    ax.hist(samples, bins=np.linspace(0.9,1.1,40), density=True)
    ax.plot(x, bkg, label=r"Lognormal $\mu=0.0, \sigma=0.02$")
    ax.axvline(samples.mean(), 0, 1, color="red", alpha=0.6, linestyle="--", label="mean: {:.2f}".format(samples.mean()))
    ax.legend(loc=1)
    ax.set_xlabel("x")    

fig, ax = plt.subplots(1, 3, figsize=(33, 12))
plt_bkg(ax[1])
plt_signal(ax[0])
plt_lumi(ax[2])
ax[0].set_title("Signal")
ax[1].set_title("Background")
ax[2].set_title("Luminosity")
plt.suptitle("The prior distributions for the inputs")
plt.tight_layout()
plt.savefig("priors.pdf")
#plt.show()
