### Importing Data and Library
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import time
data = pd.read_csv('data',delimiter="=")
data.set_index("Varibale_name", drop=True, inplace=True)
#print(data)
epsilon_0= data.loc["epsilon_0"]
alpha_ry=data.loc["alpha_ry"]
alpha_rz=data.loc["alpha_rz"]
effmass_y=data.loc["effmass_y"]
effmass_z=data.loc["effmass_z"]
alatticevector=data.loc["alatticevector"]
blatticevector=data.loc["blatticevector"]
## Contants and Variables
#planck_contant =  4.135667662e-15 ## eV.s
planck_contant = 6.62607015e-34 ## J.s
h_bar = planck_contant/(2*math.pi)
m_e = 9.011e-31 ## kg
effM_y,effM_z = m_e*effmass_y, m_e*effmass_z
## Main Equation
def E_cal(K_y,K_z):
    E1 = epsilon_0
    ## in E2 - 1e20 used to convert k to m-1, then answer is in eV. This eV is converted to J by dividing by 1.6e-19
    E2= 1/1.6e-19*1e20*((h_bar**2*K_y**2/2/effM_y)+(h_bar**2*K_z**2/2/effM_z))
    E3 = math.sqrt(alpha_ry**2*K_y**2+alpha_rz**2*K_z**2)
    return (E1+E2+E3,E1+E2-E3)
## Setting up for Plotting
X = [round(x, 3) for x in list(np.arange(-0.5, 0.52, 0.001))]
Y1 = [2*x for x in X]
Y2 = [2*x for x in X]
for i in range(len(X)):
    if X[i]<0:Y1[i],Y2[i]=E_cal(0,-X[i]*2*math.pi/blatticevector)
    else:Y1[i],Y2[i]=E_cal(X[i]*2*math.pi/alatticevector,0)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(X, Y1, c ="red")
plt.plot(X, Y2, c ="red")
plt.axhline(0, color='black', linewidth=0.5),plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('$\longleftarrow$ $K_z$   |   $K_y$ $\longrightarrow$')
plt.ylabel('E (eV)')
plt.title('E vs K')

def format_func(N, tick_number):
    if N == 0:return r"$\Gamma$"
    elif N == -0.5:return r"$Z$"
    elif N == 0.5:return r"$Y$"

plt.xticks(np.arange(-0.5, 0.6, 0.5))
ax.xaxis.set_major_formatter(tick.FuncFormatter(format_func))
date_time = time.ctime()
plt.savefig(f'plot_{date_time}.png')


# SAVING DATA
titles = ['w', 'E+', 'E-']
np.savetxt('Output1.dat', np.column_stack((X, Y1, Y2)), fmt='%f', delimiter='\t', header='\t\t\t'.join(titles))
np.savetxt('Output2.dat', np.column_stack(([x + 0.5 for x in X], Y1, Y2)), fmt='%f', delimiter='\t', header='\t\t\t'.join(titles))
