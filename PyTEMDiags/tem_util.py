# Joe Hollowed
# University of Michigan 2023
#
# Providing a set of utility functions for use by tem_diagnostics.py


# =========================================================================
    

import pdb
import numpy as np
import xarray as xr
from timeit import default_timer


# -------------------------------------------------------------------------


class logger:
    def __init__(self, debug, name='PyTEMDiags', header=False):
        self.debug = debug
        self.name  = name
        self.timer_running = False
        if(debug and header): 
            print('\n-------- {} Debug logging active ---------'.format(name))
    def print(self, s, with_timer=False, end=None):
        if(self.debug): print('({} debug) {}'.format(self.name, s), end=end)
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
        

def multiply_lat(A, lat):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(lat, xr.core.dataarray.DataArray)): 
        lat = xr.DataArray(lat)

    Alat        = A.copy(deep=True)
    Alat.values = np.einsum('ijk,i->ijk', A, lat)

    if(hasattr(A, 'name')): Alat.name = 'prod_{}_lat'.format(A.name)
    if 'long_name' in A.attrs:
        Alat.attrs['long_name'] = 'product of {} and latitude'.format(A.long_name)
    elif hasattr(A, 'name'):
        Alat.attrs['long_name'] = 'product of {} and latitude'.format(A.name)
    if 'units' in A.attrs and 'units' in lat.attrs:
        Alat.attrs['units'] = '{} {}'.format(A.units, lat.units)
    
    return Alat


def multiply_p_1d(A, p):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)

    Ap        = A.copy(deep=True)
    Ap.values = np.einsum('ijk,j->ijk', A, p)

    Ap.name = 'prod_{}_p'.format(A.name)
    
    if(hasattr(A, 'name')): Ap.name = 'prod_{}_p'.format(A.name)
    if 'long_name' in A.attrs:
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.long_name)
    elif hasattr(A, 'name'):
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        Ap.attrs['units'] = '{} {}'.format(A.units, p.units)

    return Ap


def multiply_p_3d(A, p):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)

    Ap = A*p
    
    if(hasattr(A, 'name')): Ap.name = 'prod_{}_p'.format(A.name)
    if 'long_name' in A.attrs:
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.long_name)
    elif hasattr(A, 'name'):
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        Ap.attrs['units'] = '{} {}'.format(A.units, p.units)
    
    return Ap


def lat_gradient(A, lat, logger=None):
   
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(lat, xr.core.dataarray.DataArray)): 
        lat = xr.DataArray(lat)

    dA_dlat        = A.copy(deep=True)
    dA_dlat.values = np.gradient(A, lat, axis=0)
    
    if(hasattr(A, 'name')):
        dA_dlat.name               = 'd{}_dlat'.format(A.name)
    if 'long_name' in A.attrs:
        dA_dlat.attrs['long_name'] = 'meridional derivative of {}'.format(A.attrs['long_name'])
    elif hasattr(A, 'name'):
        dA_dlat.attrs['long_name'] = 'meridional derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in lat.attrs:
        dA_dlat.attrs['units']     = '{}/{}'.format(A.units, lat.units)
    
    return dA_dlat


def p_gradient_1d(A, p, logger=None):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)

    dA_dp        = A.copy(deep=True)
    dA_dp.values = np.gradient(A, p, axis=1)
    
    if(hasattr(A, 'name')):
        dA_dp.name               = 'd{}_dp'.format(A.name)
    if 'long_name' in A.attrs:
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.attrs['long_name'])
    elif hasattr(A, 'name'):
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        dA_dp.attrs['units']     = '{}/{}'.format(A.units, p.units)
    
    return dA_dp


def p_gradient_3d(A, p, logger=None):

    # usnig notation of https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)
    
    dA_dp = A.copy(deep=True)
    nlev  = A.shape[1]

    hs = np.diff(p[:,:-1,:], axis=1)
    hd = np.diff(p[:,1:,:],  axis=1)
    fs = A[:,:-2,:].values
    fd = A[:,2:,:].values
    f  = A[:,1:-1,:].values

    forward_diff  = np.diff(A[:,0:2,:], axis=1) / np.diff(p[:,0:2,:], axis=1)
    backward_diff = np.diff(A[:,-2:,:], axis=1) / np.diff(p[:,-2:,:], axis=1)
    centered_diff = (hs**2*fd + (hd**2 - hs**2)*f - hd**2*fs) / (hs*hd*(hd+hs))
    dA_dp.values  = np.hstack([forward_diff, centered_diff, backward_diff])

    if(hasattr(A, 'name')):
        dA_dp.name               = 'd{}_dp'.format(A.name)
    if 'long_name' in A.attrs:
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.attrs['long_name'])
    elif hasattr(A, 'name'):
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        dA_dp.attrs['units']     = '{}/{}'.format(A.units, p.units)

    # verify that one column gives the same result as np.gradient
    col0    = dA_dp[0, :, 0]
    col0_np = np.gradient(A[0,:,0], p[0,:,0])
    maxdiff = np.argmax(col0.values - col0_np)
    pdiff   = col0[maxdiff]/((col0 - col0_np)[maxdiff]) * 100
    assert np.sum(np.isclose(col0, col0_np)) == nlev,\
           'vertical pressure gradient of var {} at (column,time) = (0,0) '\
           'inconsistent with np.gradient by {}%'.format(A.name, pdiff)

    return dA_dp

    # previous iterative implementation:
    #ncol, nlev, nt = A.shape[0], A.shape[1], A.shape[2]
    #dA_dp = np.zeros((ncol, nlev, nt))
    #for i in range(ncol):
    #    for t in range(nt):
    #        if(logger is not None): 
    #            logger.print('computing dAdp at (col, time) = ({}, {})'.format(i, t), end='\r')
    #        dA_dp[i, :, t] = np.gradient(A[i,:,t], p[i,:,t])
    #return dA_dp


def p_integral_1d(A, p, logger=None):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)
    
    intAdp        = A.copy(deep=True)
    intAdp.values = np.trapz(A, p, axis=1)
    
    if(hasattr(A, 'name')):
        intAdp.name               = 'int{}dp'.format(A.name)
    if 'long_name' in A.attrs:
        intAdp.attrs['long_name'] = 'vertical integral of {}'.format(A.attrs['long_name'])
    elif hasattr(A, 'name'):
        intAdp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        intAdp.attrs['units']     = '{}/{}'.format(A.units, p.units)
    
    return dA_dp


def p_integral_3d(A, p, logger=None):

    # https://en.wikipedia.org/wiki/Trapezoidal_rule
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)
     
    intAdp        = A.copy(deep=True)
    zeros         = np.zeros((A.shape[0], 1, A.shape[2]))
    trapz_bases   = np.diff(p, axis=1)
    trapz_heights = (A + np.roll(A, 1, axis=1))[:,1:,:]
    trapz         = np.cumsum(trapz_bases * 0.5*trapz_heights, axis=1)
    intAdp.values = np.hstack([zeros, trapz]) #integral at model top is from p=[0, 0]; should be 0
    
    if(hasattr(A, 'name')):
        intAdp.name               = 'int{}dp'.format(A.name)
    if 'long_name' in A.attrs:
        intAdp.attrs['long_name'] = 'vertical integral of {}'.format(A.attrs['long_name'])
    elif hasattr(A, 'name'):
        intAdp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        intAdp.attrs['units']     = '{}/{}'.format(A.units, p.units)
    
    # verify that one column gives the same total-column result as np.trapz
    col0    = intAdp[0, -1, 0] # bottom pressure level, total column integral
    col0_np = np.trapz(A[0,:,0], p[0,:,0])
    pdiff   = col0/(col0 - col0_np) * 100
    assert np.isclose(col0, col0_np),\
           'vertical pressure integral of var {} at (column,time) = (0,0) '\
           'inconsistent with np.trapz by {}%'.format(A.name, pdiff)
    
    return intAdp
    
    # previous iterative implementation:
    #ncol, nlev, nt = A.shape[0], A.shape[1], A.shape[2]
    #intAdp = np.zeros((ncol, nlev, nt))
    #for i in range(ncol):
    #    for t in range(nt):
    #        if(logger is not None): 
    #            logger.print('computing int(Adp) at (col, time) = ({}, {})'.format(i, t), end='\r')
    #        intAdp[i, :, t] = np.trapz(A[i,:,t], p[i,:,t])
    #return intAdp

