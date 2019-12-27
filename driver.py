# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:10:58 2019

@author: hplgit / James
@source: https://github.com/hplgit/bumpy
"""
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from generate_road_profiles import generate_bumpy_roads
from bumpy_model import bumpy_model


if __name__ == '__main__':
    
    # Input parameters as arguments here.
    # Assume units in M.K.S.
    generate_bumpy_roads(L=500, nroads=10, resolution=200)
    data = bumpy_model(m=60, b=80, k=60, v=5)
    
    # Root mean square values
    u_rms = [np.sqrt((1./len(u))*np.sum(u**2))
             for h, F, u in data[2:]]
    print ('u_rms:', u_rms)
    print ('Simulated for t in [0,%g]' % data[1][-1])

    # Save data list to file
    outfile = open('bumpy.res', 'wb')
    cPickle.dump(data, outfile)
    outfile.close()

    # Post-processing
    outfile = open('bumpy.res', 'rb')
    data = cPickle.load(outfile)
    outfile.close()

    # data = [x, t, [h, a, u], [h, a, u], ..., u_rms]
    x, t = data[0:2]
    
    # ==============================
    # Post-processing
    # Plot u for second realization
    realization = 1
    u = data[2+realization][2][:]
    plt.plot(t, u)
    plt.title('Displacement')

    
    # Compute and plot velocity in second realization
    dt = t[1] - t[0]
    v = np.zeros_like(u)
    v[1:-1] = (u[2:] - u[:-2])/(2*dt)
    v[0] = (u[1] - u[0])/dt
    v[-1] = (u[-1] - u[-2])/dt
    plt.figure()
    plt.plot(t, v)
    plt.legend(['velocity'])
    plt.xlabel('t')
    plt.title('Velocity')
    
    
    # Smooth the velocity (only internal points)
    v[1:-1] = (v[2:] + 2*v[1:-1] + v[:-2])/4.0
    plt.figure()
    plt.plot(t, v)
    plt.legend(['smoothed velocity'])
    plt.xlabel('t')
    plt.title('Velocity')
    
    plt.show()
