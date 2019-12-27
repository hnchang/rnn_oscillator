# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:16:48 2019

@author: hplgit / jct
"""

import sys
import numpy as np


def bumpy_model(m=60, b=80, k=60, v=5):
    """
    Solve model for verticle vehicle vibrations.

    =========   ==============================================
    variable    description
    =========   ==============================================
    m           mass of system
    b           friction parameter
    k           spring parameter
    v           (constant) velocity of vehicle
    Return      data (list) holding input and output data
                [x, t, [h,a,u], [h,a,u], ...]
    =========   ==============================================   
"""
    filename = 'bumpy.txt'
    try:
        h_data = np.loadtxt(filename)  # read numpy array from file
    except ValueError:
        print ('Loading file error')
        sys.exit(1)

    x = h_data[0,:]                # 1st column: x coordinates
    h_data = h_data[1:,:]          # other columns: h shapes

    t = x/v                        # time corresponding to x
    dt = t[1] - t[0]
    
    if dt > 2/np.sqrt(k/float(m)):
        print ('Unstable scheme')

    from solver import solver
    
    def s(u):
            return k*u

    data = [x, t]      # key input and output data (arrays)
    for i in range(h_data.shape[0]):
        h = h_data[i,:]            # extract a column
        # a = acceleration(h, x, v)
        a = acceleration_vectorized(h, x, v)
        F = -m*a

        u, t = solver(I=0, V=0, m=m, b=b, s=s, F=F, t=t, damping='linear')
        data.append([h, F, u])
    return data    


def acceleration(h, x, v):
    """Compute 2nd-order derivative of h."""
    # Method: standard finite difference aproximation
    d2h = np.zeros(h.size)
    dx = x[1] - x[0]
    for i in range(1, h.size-1, 1):
        d2h[i] = (h[i-1] - 2*h[i] + h[i+1])/dx**2
    # Extraplolate end values from first interior value
    d2h[0] = d2h[1]
    d2h[-1] = d2h[-2]
    a = d2h*v**2
    return a

def acceleration_vectorized(h, x, v):
    """Compute 2nd-order derivative of h. Vectorized version."""
    d2h = np.zeros(h.size)
    dx = x[1] - x[0]
    d2h[1:-1] = (h[:-2] - 2*h[1:-1] + h[2:])/dx**2
    # Extraplolate end values from first interior value
    d2h[0] = d2h[1]
    d2h[-1] = d2h[-2]
    a = d2h*v**2
    return a
