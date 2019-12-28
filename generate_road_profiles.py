# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:48:42 2019

@author: hplgit / jct
"""
import numpy as np
from solver import solver

##### =============== Generate (a) bumpy road(s) ===============

# Simple version
amps =[1.0, 5.0, 10.0];
freqs = [0.1, 0.5, 1.2];
#phis = [0.0, np.pi/3.0, np.pi/2];
gau_rel_centers = [0.25, 0.5, 0.75]
gau_rel_spans = [0.25, 0.5, 0.75]


def generate_bumpy_road(nbumps=10, L=200, resolution=200):
    """Randomly pick an element from an array of parameters of details"""
    #amp = np.random.choice(amps)
    #freq = np.random.choice(freqs)
    #phi = np.random.choice(phis)
    #gau_rel_center = np.random.choice(gau_rel_centers)
    #gau_rel_span = np.random.choice(gau_rel_spans) 
    
    
    """Generate one road profile by using a vibration ODE."""
    n = nbumps*resolution      # no of compute intervals along the road
    x = np.linspace(0, L, n+1) # points along the road
    dx = x[1] - x[0]           # step along the road
    
    white_noise = np.random.randn(n+1)# /np.sqrt(dx)
    # sinusoidal = amp*np.sin(freq*x/len(x) ) 
    #gau_modul = np.exp(-(((x - L*gau_rel_center)/(gau_rel_span*L))**2 / 2))
    
    f = white_noise#*gau_modul #+ white_noise
    
    # Compute h(x)
    k = 1.
    m = 4.
    if dx > 2/np.sqrt(k/m):
        print ('Unstable scheme')
    def s(u):
        return k*u

    h, x = solver(I=0, V=0, m=m, b=5, s=s, F=f, t=x, damping='linear')
    # Normalization
    h = h/h.max()*0.2
    return h, x

def generate_bumpy_roads(L, nroads, resolution):
    
    """Generate many road profiles."""
    # np.random.seed(1)
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
    generate_bumpy_roads(L=500, nroads=200, resolution=20)



