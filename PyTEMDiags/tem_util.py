# Joe Hollowed
# University of Michigan 2023
#
# Providing a set of utility functions for use by tem_diagnostics.py


# =========================================================================
    

import numpy as np
from timeit import default_timer


# -------------------------------------------------------------------------


class logger:
    def __init__(self, debug, name='PyTEMDiags', header=False):
        self.debug = debug
        self.name  = name
        self.timer_running = False
        if(debug and header): 
            print('\n-------- {} Debug logging active ---------'.format(name))
    def print(self, s, with_timer=False):
        if(self.debug): print('({} debug) {}'.format(self.name, s))
        if(with_timer):
            self.timer()
    def timer(self, start_silent=True):
        if not self.timer_running:
            self.start = default_timer()
            self.timer_running = True
            if not start_silent: self.print('timer started')
        else:
            self.stop = default_timer()
            self.timer_running = False
            self.print('elapsed time: {:.2f} seconds'.format(self.stop - self.start))

def lat_gradient(A, lat):
    return np.gradient(A, lat, axis=0)

def p_gradient_1d(A, p):

def p_gradient_3d(A, p):
    ncol, nlev, nt = A.shape[0], A.shape[1], A.shape[2]
    dAdp = np.zeros((ncol, nlev, nt))
    for i in range(ncol):
        for t in range(nt):
            dAdp[i, :, t] = np.gradient(A[i,:,t], p[i,:,t])
    return dAdp
