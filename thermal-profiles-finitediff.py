#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:36:09 2018

Python2 code for Landscape Modeling class exercise 3 - Part 2
Finite difference method using forward in time and centered in space

@author: nadine
"""
#%% PART 2 - FINITE DIFF SOLUTION

import numpy as np
import matplotlib.pyplot as plt
import os

degree = u"\u00b0" # unicode symbol for degree for labeling plots nicely


#%% Try with explicit equation

## define variables - simply for starters
#Qm = 45 * .001                       # heat flux in W/m^2,  mantle heat flux = bottom boundary condition
#k = 2.5                             # thermal conductivity in W/(m-K)
#kappa = 1 * 1e-6                    # thermal diffusivity in m^2/sec
#period = 60*60*24*365.25            # one year of seconds
#
#dz = 0.1                            # meter, try making this smaller
#zmax = 1.                          # meters
#z = np.arange(0.,zmax+dz,dz)         # depth array in meters
#
#years = 1.                          # number of years to run for
#dt = (period/365.)* .01                    # one day of seconds
#tmax = period * years               # years
#t = np.arange(0.,tmax+dt,dt)   # time array in days
#
#T = np.ndarray(shape=(len(t),len(z)), dtype=float)
#
#plt.axvline(0,0,15,color='k',linestyle='--') # plot line at X = 0
##plt.ylim(0,15)
##plt.xlim(-25,5,5)
#plt.gca().invert_yaxis()                    # invert Y axis
#plt.xlabel('Temperature ('+degree+'C)')
#plt.ylabel('Depth (m)')
#
## BOUNDARY CONDITIONS --> bottom Q and top T
## try smaller time steps
## is the indexing correct for plotting??
#
#for n in range(len(t)-1):
#    for i in range(len(z)-1):
#        T[(n+1),i] = T[n,i] + (dt*kappa)*((T[n,(i+1)]-(2*T[n,i])+T[n,(i-1)])/(dz**2))
#    if n % 10 == 0:
#        plt.plot(T[n,:],z,linewidth=0.5)

#%% Try again with other method (dXdy arrays)

# set variables
years = 1                  # number of years to run for
plots = years*100           # plot every "plots" timestep geotherm

k = 2.5                     # thermal conductivity in W/(m-K)
kappa = 1e-6                # thermal diffusivity in m^2/sec 
rhoc = k/kappa              # calculate rho*c based on values of k and kappa
period = 60*60*24*365.25    # one year of seconds
Qm = 45 * .001              # heat flux in W/m^2,  mantle heat flux = bottom boundary condition
MAT = -10                   # mean annual temp at the surface in celcius
dT = 15                     # annual +/- change in temp in celcius

dt = (period / 365 ) * .01  # timestep: one day times a multiplier to keep dt small
tmax = period * years       # max time = 1 year * number of years set above
t = np.arange(0,tmax,dt)    # time array 

dz = 0.1                    # depth step in meters
zmax = 20.0                 # max depth
z = np.arange(0,zmax+dz,dz) # depth array

#%% second try run loop with dXdy arrays

os.system("rm tmp*.png") # remove any tmp.png files so don't get erroneous frames in our movie

# initialize arrays 
Ts = np.zeros(len(t)) # surface temp array
T = np.ndarray(shape=(len(z),len(t)), dtype=float) # temp/geotherm array
Q = np.ndarray(shape=(len(z),len(t)), dtype=float) # Q / heat flux array
dTdz = np.ndarray(shape=(len(z)-1,len(t)), dtype=float) 
dQdz = np.ndarray(shape=(len(z)-1,len(t)), dtype=float)
dTdt = np.ndarray(shape=(len(z)-1,len(t)), dtype=float)

Q[-1,:] = Qm # apply mantle heat flux (Qm) as bottom boundary condition for all time

# set up plot for movie
plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,20)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')

   
# finite diff loop through all time t
for i in range(len(t)-1):
   Ts[i] = MAT + dT * np.sin(((2*np.pi*t[i]))/period) # calculate surface temperature (Ts) for time i
   T[0,i] = Ts[i]  # apply surface temp as top boundary condition (z = 0) for time i
   dTdz[:,i] = np.diff(T[:,i]) / dz # calculate T gradient with depth
   Q[:-1,i] = -k * dTdz[:,i] # calculate heat content at every depth
   dQdz[:,i] = np.diff(Q[:,i]) / dz # heat flow gradient
   dTdt[:,i] = (-1/rhoc) * dQdz[:,i] # temp gradient with time
   T[1:,i+1] = T[1:,i] + (dTdt[:,i]*dt) # write T 
   if i % plots == 0: # plot at every plots timestep
       plt.plot(T[:,i],z) # plot T for all depth at time i
       plt.text(-20, 12.5, 'time (days):'+ str((i/100))) # add label for total time = 
       plt.savefig('tmp'+str(i/plots)+'.png',bbox_inches="tight",dpi=150) # save plot for movie


# make a movie with ffmpeg!
#fps = 35
#os.system("rm movie.mp4") # remove a previous movie.mp4 file so don't get overwrite problems
#os.system("ffmpeg -r 35 -pattern_type sequence -i tmp'%d'.png -vcodec mpeg4 movie.mp4") 
#os.system("rm tmp*.png")

#%% first try run loop with dXdy arrays 
# starts ok for first .01 of a year, then goes crazy

#num = 50


#z = np.linspace(0,zmax+dz,num)
z = np.arange(0,zmax+dz,dz)
#Q = np.zeros(num)
Q = np.zeros(len(z))
Q[-1] = Qm # bottom boundary condition, Qm = mantle heat flow

dt = (period / 365 ) * .01  # timestep: one day times a multiplier to keep dt small
tmax = period * years       # max time = 1 year * number of years set above
t = np.arange(0,0.01*tmax,dt)

#T = np.ones(num)
T = np.ones(len(z))
#T = np.ndarray(shape=(len(z),len(t)), dtype=float)

#dTdz = np.zeros(num-1)
#dQdz = np.zeros(num-1)
#dTdt = np.zeros(num-1)

dTdz = np.zeros(len(z)-1)
dQdz = np.zeros(len(z)-1)
dTdt = np.zeros(len(z)-1)

for n in range(len(t)):
    for i in range(len(z)):
        Ts = MAT + dT * np.sin(((2*np.pi*t[n]))/period) # sinusoidal equation for surface temp
        T[0] = Ts                                       # top boundary condition
        dTdz = np.diff(T) / dz
        Q[:-1] = -k * dTdz
        dQdz = np.diff(Q) / dz
        dTdt = -(1/rhoc) * dQdz
        T[1:] = T[1:] + (dTdt * dt) 
        if n % 1000 == 0:
            plt.plot(T,z)

plt.axvline(0,0,15,color='k',linestyle='--')
plt.ylim(0,20)
plt.xlim(-25,5,5)
plt.gca().invert_yaxis()
plt.xlabel('Temperature ('+degree+'C)')
plt.ylabel('Depth (m)')
plt.show() 


#%% Myelene's loop code

#for i in range(0,len(time)-1):
#   T[0,i] = T_bound[i] # make boundary condition temperature surface temperature for time i
#   Q[-1,i] = 30e-3  # geothermal gradient in degrees C per meter
#   dTdz[:,i] = np.divide(np.diff(T[:,i]),dz) # calculate T gradient with depth
#   Q[:-1,i] = -k*dTdz[:,i] # calculate heat content at every depth
#   dQdz[:,i] = np.divide(np.diff(Q[:,i]),dz) # heat flow gradient
#   dTdt[:,i] = np.multiply(-1/(roh*c),dQdz[:,i]) # temp gradient with time
#   T[1:,i+1] = np.add(T[1:,i],dTdt[:,i]*dt) # write T 
