### Importing Data and Library
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import time

band_dic = {}
bands = []
tmp_list = []
with open("conduction_band.dat", "r") as file:
    for line in file:
        val = line.split()
        if val == []:
            band_dic[band_name] = np.array([tmp_list]).transpose()
            continue
        if val[0] == "#":
            tmp_list = []
            band_name = val[1] + " " + val[2]
            bands.append(band_name)
            continue
        tmp_list.append([float(val[0]), float(val[1])])
band_dic[band_name] = np.array([tmp_list]).transpose()

for x in bands:
    tmp = band_dic[x]
    plt.plot(tmp[0], tmp[1], c="red")

# print(band_dic)
plt.savefig(f"raw_data.png")


def firstorderdiff(k, band):
    array_1 = []
    tmp = 0
    sign = ""
    last_value = None
    for x in range(len(band)):
        if x == 0:
            continue
        if x == len(band):
            continue
        if k[x] - k[x - 1] == 0:
            continue
        if band[x] - band[x - 1] == 0:
            continue
        if band[x] - band[x - 1] < 0:
            sign_tmp = "positive"
        else:
            sign_tmp = "negative"
        if sign != sign_tmp:
            if sign == "":
                sign = sign_tmp
                continue
            else:
                array_1.append(k[x - 1])
                sign = sign_tmp
                continue
    return array_1


x = "Band-Index 153"
tmp = band_dic[x]
peaks = firstorderdiff(tmp[0], tmp[1])
print(peaks)
for x in peaks:
    plt.axvline(x, color="black", linewidth=0.5)

x = "Band-Index 154"
tmp = band_dic[x]
peaks = firstorderdiff(tmp[0], tmp[1])
print(peaks)
for x in peaks:
    plt.axvline(x, color="yellow", linewidth=0.5)


plt.savefig(f"raw_marked_peaks.png")
