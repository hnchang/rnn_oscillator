# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:48:42 2019

@author: hplgit / jct
"""
import numpy as np
from solver import solver

##### =============== Generate (a) bumpy road(s) ===============

def generate_bumpy_road(nbumps=10, L=200, resolution=200):
    """Generate one road profile by using a vibration ODE."""
    n = nbumps*resolution      # no of compute intervals along the road
    x = np.linspace(0, L, n+1) # points along the road
    dx = x[1] - x[0]           # step along the road
    white_noise = np.random.randn(n+1)/np.sqrt(dx)
    # Compute h(x)
    k = 1.
    m = 4.
    if dx > 2/np.sqrt(k/m):
        print ('Unstable scheme')
    def s(u):
        return k*u

    h, x = solver(I=0, V=0, m=m, b=3, s=s, F=white_noise,
                  t=x, damping='linear')
    h = h/h.max()*0.2
    return h, x

def generate_bumpy_roads(L, nroads, resolution):
    
    """Generate many road profiles."""
    np.random.seed(1)
    nbumps = int(L/25.)
    h_list = []
    for i in range(nroads):
        h, x = generate_bumpy_road(nbumps, L, resolution)
        h_list.append(h)
    h_list.insert(0, x)
    data = np.array(h_list)
    np.savetxt('bumpy.txt', data)  # saves in gzip'ed format



if __name__ == '__main__':
    
    # Test the function: 
    generate_bumpy_roads(L=500, nroads=10, resolution=200)



