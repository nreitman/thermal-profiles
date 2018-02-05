# -*- coding: utf-8 -*-
"""
Python2 code for Landscape Modeling class exercise 3
Thermal Profiles (AKA, snake writhing in an exponential funnel)
Written by Nadine Reitman on 2/2/2018
"""
#%% import modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

degree = u"\u00b0" # unicode symbol for degree for labeling plots nicely

#%% PART 1A - Steady state solution for the geotherm in the top 800 m

# initialize variables
MAT = -12   # mean annual temp in degrees C
k = 2.5     # thermal conductivity in W/(m-K)
Q = 45      # heat flux in mW/m^2
Q = 45 * .001 # heat flux in W/m^2
z = np.linspace(0,800,num=801)

# caluclate geotherm temps
T = MAT + ((Q/k) * z)

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
        T[i] = T[30] + ((Q/k) * (z[i]-z[30]))

# plot
plt.axvline(0,0,800,color='k',linestyle='--')
plt.plot(T,z,'r')
plt.ylim(0,800)
plt.xlim(-15,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST - now with min/max bounds

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
time = np.linspace(1,period,num=30) # 1 year in 30 time steps
T = np.ndarray(shape=(30,30), dtype=float) 

    
# calculate geotherms
for i in range(len(time)):
    T[i] = MAT + (dT * np.exp(-z/zstar)) * np.sin(((2*np.pi*time[i])/period)-(z/zstar))
    plt.plot(T[i],z,linewidth=0.5)


Tmin = MAT - (dT * np.exp(-z/zstar))
Tmax = MAT + (dT * np.exp(-z/zstar))

# plot
plt.axvline(0,0,15,color='k',linestyle='--')
plt.plot(Tmin,z,'k')
plt.plot(Tmax,z,'k')
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST - plotting min/max temp bounds only

Tmin = MAT - (dT * np.exp(-z/zstar))
Tmax = MAT + (dT * np.exp(-z/zstar))

plt.plot(Tmin,z,'k')
plt.plot(Tmax,z,'k')

plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show()

#%% PART 1C - Sinusoidal GST - time-temp history for 10 depths

#%% PART 1C - Sinusoidal GST - daily variation

#%% PART 1C - Sinusoidal GST - animated!

#%% PART 2 - FINITE DIFF SOLUTION

Q = 45 * .001                       # heat flux in W/m^2,  mantle heat flux = bottom boundary condition
k = 2.5                             # thermal conductivity in W/(m-K)
kappa = 1 * 1e-6                    # thermal diffusivity in m^2/sec
period = 60*60*24*365.25            # one year of seconds

dz = 0.1                            # meter, try making this smaller
zmax = 20                           # meters
z = np.arange(0,zmax+1,dz)          # depth array in meters

years = 3                           # number of years to run for
dt = period/365.                    # one day of seconds
tmax = period * years               # years
time = np.arange(0,(tmax+dt),dt)    # time array in days



