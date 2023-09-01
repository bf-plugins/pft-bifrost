#!/usr/bin/env python
import os,sys, glob
import numpy as np
from matplotlib import pyplot as plt

bandpassfiles = sorted(glob.glob("bandpass.bf*txt"))

inps = []
for i in range(8):
    b = "bandpass.bf{0:02d}.txt".format(i)
    if not os.path.exists(b):
        print("No file",b)
        sys.exit()
    inps.append(np.loadtxt(b, delimiter=','))
data = np.concatenate((inps)).transpose()
for i in range(data.shape[0]-1):
    plt.plot(data[0], np.log(data[i+1]), label="Input " + str(i))
    print ("For input", i, "median log amplitude was", np.median( np.log(data[i+1])))
plt.title("Log Amplitude")
plt.savefig("Bandpasses.png")
for module in range((data.shape[0]-1)//12):
    plt.clf()
    for cassette in range(12):
        i = 12*module + cassette
        plt.plot(data[0], np.log(data[i+1]), label="Module " + str(module) + ":" + str(cassette))
    plt.savefig("module.{0:02d}.png".format(module))

