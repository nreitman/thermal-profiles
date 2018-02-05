# -*- coding: utf-8 -*-
"""
Python2 code for Landscape Modeling class exercise 3
Thermal Profiles (AKA, snake writhing in an exponential funnel)
Written by Nadine Reitman on 2/2/2018
"""
#%% import modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

degree = u"\u00b0"

#%% PART 1A - Steady state solution for the geotherm in the top 800 m

# initialize variables
MAT = -12   # mean annual temp in degrees C
k = 2.5     # thermal conductivity in W/(m-K)
Q = 45      # heat flux in mW/m^2
Q = 45 * .001 # heat flux in W/m^2
z = np.linspace(0,800,num=801)

# caluclate geotherm temps
T = MAT + ((Q/k) * depth)

# plot
plt.axvline(0,0,800,color='k',linestyle='--')
plt.plot(T,z,'r')
plt.ylim(0,800)
plt.xlim(-15,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1B - decrease the thermal conductivity in the top 30 m to 1.2 W/(m-K)

# initialize variables
MAT = -12   # mean annual temp in degrees C
k = 2.5     # thermal conductivity in W/(m-K)
Q = 45      # heat flux in mW/m^2
Q = 45 * .001 # heat flux in W/m^2
z = np.linspace(0,800,num=801)

# caluclate geotherm temps
for i in range(len(z)):
    if z[i] <= 30:
        k = 1.2
        T[i] = MAT + ((Q/k) * z[i])
    elif z[i] > 30:
        k = 2.5
        T[i] = MAT + ((Q/k) * z[i])

# plot
plt.axvline(0,0,800,color='k',linestyle='--')
plt.plot(T,z,'r')
plt.ylim(0,800)
plt.xlim(-15,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST

# initialize variables
MAT = -10   # mean annual temp in degrees C
dT = 15     # change in temp in degrees C
#k = 2.5     # thermal conductivity in W/(m-K)
k = 1.2     # because we're still in top 30m
#kappa = 1   # thermal diffusivity in mm^2/sec
kappa = 1 * 1e-6 # thermal diffusivity in m^2/sec
period = 3.154*1e7
zstar = np.sqrt((kappa * period) / np.pi)
z = np.linspace(0,15,num=30) # depth, top 15 m
time = np.linspace(1,period,num=30)
T = np.ndarray(shape=(30,30), dtype=float) 
    
# caluclate geotherms
for i in range(len(time)):
    T[i] = MAT + (dT * np.exp(-z/zstar)) * np.sin(((2*np.pi*time[i])/period)-(z/zstar))
    plt.plot(T[i],z,linewidth=0.5)

# plot
plt.axvline(0,0,15,color='k',linestyle='--')
plt.plot(T[21],z,color='black',linewidth=1.0)
plt.plot(T[7],z,color='black',linewidth=1.0)
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST - min/max temp bounds

Tmin = MAT - (dT * np.exp(z/zstar))
Tmax = MAT + (dT * np.exp(z/zstar))

# plot
plt.plot(T[7],z,color='grey',linewidth=1.0)
plt.plot(T[8],z,color='yellow',linewidth=1.0)
plt.plot(T[9],z,color='orange',linewidth=1.0)
plt.plot(T[10],z,color='red',linewidth=1.0)
plt.plot(T[24],z,color='blue',linewidth=1.0)
plt.plot(T[23],z,color='green',linewidth=1.0)
plt.plot(T[22],z,color='purple',linewidth=1.0)
plt.plot(T[21],z,color='black',linewidth=1.0)

plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST - time-temp history for 10 depths


