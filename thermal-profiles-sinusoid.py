# -*- coding: utf-8 -*-
"""
Python2 code for Landscape Modeling class exercise 3
Thermal Profiles (AKA, snake writhing in an exponential funnel)
Written by Nadine Reitman on 2/2/2018
"""
#%% import modules

import numpy as np
import matplotlib.pyplot as plt
import os

degree = u"\u00b0" # unicode symbol for degree for labeling plots nicely

#%% PART 1A - Steady state solution for the geotherm

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
plt.savefig('1a.png',bbox_inches="tight")
plt.show()

#%% PART 1B - decrease the thermal conductivity in the top 30 m

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
plt.savefig('1b.png',bbox_inches="tight")
plt.show()



#%% PART 1C - Sinusoidal GST - with min/max temperature bounds...ANIMATED!

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

# calculate temperature min/max bounds
Tmin = MAT - (dT * np.exp(-z/zstar))
Tmax = MAT + (dT * np.exp(-z/zstar))

# set up plot
plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.plot(Tmin,z,'k')
plt.plot(Tmax,z,'k')
plt.savefig('tmp1.png',bbox_inches="tight",dpi=300)

plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,15)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')

    
# calculate geotherms
for i in range(len(time)):
    T[i] = MAT + (dT * np.exp(-z/zstar)) * np.sin(((2*np.pi*time[i])/period)-(z/zstar))
    plt.plot(T[i],z,linewidth=0.5)
    plt.savefig('tmp'+str(i+2)+'.png',bbox_inches="tight",dpi=300)

# plot
plt.plot(Tmin,z,'k')
plt.plot(Tmax,z,'k')
plt.savefig('tmp'+str(i+3)+'.png',bbox_inches="tight",dpi=300)
plt.show()

#%%
# make a movie with ffmpeg!

#fps = 5
os.system("rm movie.mp4")
#os.system("ffmpeg -r 5 -pattern_type glob -i 'tmp*.png' -vcodec mpeg4 movie.mp4") # reads wildcards but not in numerical order (i.e., 0,1,10,11...19,2,20,21 etc)
os.system("ffmpeg -r 5 -pattern_type sequence -i tmp'%d'.png -vcodec mpeg4 movie.mp4")
os.system("rm tmp*.png")

#%% PART 1C - Sinusoidal GST - time-temp history for 10 depths

#%% PART 1C - Sinusoidal GST - daily variation



