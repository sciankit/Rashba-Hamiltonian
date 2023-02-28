### Importing Data and Library
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import time
import csv

band_dic = {}
bands = []
tmp_list = []

latticevector = (5.729, 5.613)


## Reading the data from the file
with open("conduction_band.dat", "r") as file:
    for line in file:
        val = line.split()
        if val == []:
            tmp_list.sort(key=lambda x: x[0])
            band_dic[band_name] = np.array([tmp_list]).transpose()
            continue
        if val[0] == "#":
            tmp_list = []
            band_name = val[1] + " " + val[2]
            bands.append(band_name)
            continue
        tmp_list.append([float(val[0]), float(val[1])])
tmp_list.sort(key=lambda x: x[0])
band_dic[band_name] = np.array([tmp_list]).transpose()

## DATA STORED IN BANDS, Now plotted
for x in bands:
    tmp = band_dic[x]
    plt.plot(tmp[0], tmp[1], c="red")

# print(band_dic)
plt.savefig(f"raw_data.png")


def E_cal(variable_cal, K):
    # print(f"ENERGY VAR {variable}")
    epsilon_0 = variable_cal[0]
    alpha_r = variable_cal[1]
    effmass = m_e * variable_cal[2]
    # print(K)
    K = (K - 0.39456) * 2 * math.pi / 5.65

    E1 = epsilon_0
    ## in E2 - 1e20 used to convert to m-1, then answer is in eV. This eV is converted to J by dividing by 1.6e-19
    E2 = 1 / 1.6e-19 * 1e20 * (h_bar**2 * K**2 / 2 / effmass)
    E3 = alpha_r * K
    return (E1 + E2 + E3, E1 + E2 - E3)


def error(varibale_err, K, Y):
    Y_out = E_cal(varibale_err, K)
    error = (Y[0] - Y_out[0]) ** 2 + (Y[1] - Y_out[1]) ** 2
    return error


def error_function(variable_ef, K_set, Y_set):
    total_error = 0
    for i in range(len(K_set)):
        total_error = total_error + error(
            variable_ef, K_set[i], [Y_set[0][i], Y_set[1][i]]
        )
    # print(total_error)
    return total_error


def gradient(variable_grad, K, Y):
    error_initial = error_function(variable_grad, K, Y)
    # print(f"Current error {error_initial}")
    grad = []
    delta = 0.0001
    a_val = float(variable_grad[0])
    b_val = float(variable_grad[1])
    c_val = float(variable_grad[2])
    ErrorWithChange = error_function([a_val + delta, b_val, c_val], K, Y)

    grad.append(ErrorWithChange - error_initial)
    errorWithChange = error_function([a_val, b_val + delta, c_val], K, Y)
    grad.append(ErrorWithChange - error_initial)
    errorWithChange = error_function([a_val, b_val, c_val + delta], K, Y)
    grad.append(ErrorWithChange - error_initial)
    return grad


def update_variable(variable_upd, K, Y):
    grad = gradient(variable_upd, K, Y)
    # print("GRAD ", grad)
    # print("Values",variable_upd)
    # print(learning_rate)
    for i in range(len(variable_upd)):
        delta = grad[i]
        # print(delta)
        delta = delta * learning_rate
        # print(delta)
        # print("var ", variable_upd[i])
        variable_upd[i] = variable_upd[i] - delta
        # print("var_up",variable_upd[i])
    return variable_upd


## Finding Minima and Maxima of graph
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
# print(peaks)
for x in peaks:
    plt.axvline(x, color="black", linewidth=0.5)

x = "Band-Index 154"
tmp = band_dic[x]
peaks = firstorderdiff(tmp[0], tmp[1])
# print(peaks)
for x in peaks:
    plt.axvline(x, color="yellow", linewidth=0.5)


plt.savefig(f"raw_marked_peaks.png")

ranges = {}
ranges[0] = (0.27, 0.39456)
ranges[1] = (0.39456, 0.48)
# print("THIS IS TMP NEW",band_dic["Band-Index 153"][0])

# print(ranges)
loc = []
for x in ranges:
    val = ranges[x]  ## x is the key, val stores the value corresponding to that key
    tmp = band_dic["Band-Index 153"][0]
    # print("THIS IS TMP",tmp)
    start_index = 0
    end_index = 0
    ran = False
    for i, value in enumerate(tmp):
        if value >= val[0] and not ran:
            start_index = i
            ran = True
        if value > val[1]:
            end_index = i - 1
            break
    loc.append((start_index, end_index))

# print("these are location",loc)

planck_contant = 6.62607015e-34  ## J.s
h_bar = planck_contant / (2 * math.pi)
m_e = 9.011e-31  ## kg
# print("THESE ARE BAND NAMES",bands)
# print("THESE ARE BAND DAT")


plt.clf()
time_step = 10
learning_rate = 0.1


for val in loc:
    # variable=[0.220777,0.07,1.7]
    variable = [0.220777, 0.5, 0.5]
    K_set = []
    Y_set = []

    plt.clf()
    KEY = ""
    for key in band_dic:
        K_set = band_dic[key][0][val[0] : val[1]]
        # print(K_set)
        break
    for name, values2 in band_dic.items():
        Y_set.append(values2[1][val[0] : val[1]])
        plt.plot(K_set, values2[1][val[0] : val[1]])

    print(f"THIS IS THE Initial {variable}")
    print(f"This is intial square error {error_function(variable,K_set,Y_set)}")

    for i in range(time_step):
        # print(f"iteratin no. {i} Variable value {variable}")
        variable = update_variable(variable, K_set, Y_set)
    print(f"THIS IS THE FINAL {variable}")
    print(f"This is square error {error_function(variable,K_set,Y_set)}")

    # variable=[0.220777,0.07,1.7]
    for tmp in K_set:
        output = E_cal(variable, tmp)
        Eng1 = output[0]
        Eng2 = output[1]
        plt.scatter(tmp, Eng1, c="b")
        plt.scatter(tmp, Eng2, c="b")

    plt.savefig(f"{val[0]}-{val[1]}.png")

    with open(f"data{val[0]}.csv", "w", newline="") as fl:
        for i in range(len(K_set)):
            fl.write(
                f"{float(K_set[i])}    {float(Y_set[0][i])}   {float(Y_set[1][i])} \n"
            )


# plt.show()
