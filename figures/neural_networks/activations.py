import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3,3, 0.01)

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def ReLU(x):
    res = x.copy()
    res[(x<0)] = 0
    return res

def SeLU(x, l=0.5, a=1):
    res = x.copy()
    sel = x>0
    res[sel] = l*x[sel]
    res[~sel] = l*a*(np.exp(x[~sel])-1)
    return res

plt.figure(figsize=(8,6))
plt.title("Most commonly used activation functions")
plt.xlabel("x")
plt.ylabel(r"g(x)")
plt.axvline(0, 0, 1, color="black", alpha=0.7)
plt.axhline(0, 0, 1, color="black", alpha=0.7)
plt.plot(x, ReLU(x), label="ReLU")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, np.tanh(x), label="tanh")
plt.plot(x, SeLU(x), label=r"SeLU ($\lambda=0.5, \alpha=1$)")
plt.ylim((-1, 2))
plt.grid()
plt.legend()
plt.savefig("activations.png", dpi=600)
plt.show()
