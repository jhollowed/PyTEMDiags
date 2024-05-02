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
    
# --------------------------------------------------        

def multiply_lat(A, lat):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(lat, xr.core.dataarray.DataArray)): 
        lat = xr.DataArray(lat)

    Alat        = A.copy(deep=True)
    Alat.values = np.einsum('ijk,i->ijk', A, lat)

    if(A.name is not None): Alat.name = 'prod_{}_lat'.format(A.name)
    if 'long_name' in A.attrs:
        Alat.attrs['long_name'] = 'product of {} and latitude'.format(A.long_name)
    elif A.name is not None:
        Alat.attrs['long_name'] = 'product of {} and latitude'.format(A.name)
    if 'units' in A.attrs and 'units' in lat.attrs:
        Alat.attrs['units'] = '{} {}'.format(A.units, lat.units)
    
    return Alat

# --------------------------------------------------        

def multiply_p(A, p):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)

    Ap        = A.copy(deep=True)
    Ap.values = np.einsum('ijk,j->ijk', A, p)

    if(A.name is not None): Ap.name = 'prod_{}_p'.format(A.name)
    if 'long_name' in A.attrs:
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.long_name)
    elif A.name is not None:
        Ap.attrs['long_name'] = 'product of {} and pressure'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        Ap.attrs['units'] = '{} {}'.format(A.units, p.units)

    return Ap

# --------------------------------------------------        

def lat_gradient(A, lat, logger=None):
   
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(lat, xr.core.dataarray.DataArray)): 
        lat = xr.DataArray(lat)

    dA_dlat        = A.copy(deep=True)
    dA_dlat.values = np.gradient(A, lat, axis=0)
    
    if(A.name is not None):
        dA_dlat.name               = 'd{}_dlat'.format(A.name)
    if 'long_name' in A.attrs:
        dA_dlat.attrs['long_name'] = 'meridional derivative of {}'.format(A.attrs['long_name'])
    elif A.name is not None:
        dA_dlat.attrs['long_name'] = 'meridional derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in lat.attrs:
        dA_dlat.attrs['units']     = '{}/{}'.format(A.units, lat.units)
    
    return dA_dlat

# --------------------------------------------------        

def p_gradient(A, p, logger=None):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)

    dA_dp        = A.copy(deep=True)
    dA_dp.values = np.gradient(A, p, axis=1)
    
    if(A.name is not None):
        dA_dp.name               = 'd{}_dp'.format(A.name)
    if 'long_name' in A.attrs:
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.attrs['long_name'])
    elif A.name is not None:
        dA_dp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        dA_dp.attrs['units']     = '{}/{}'.format(A.units, p.units)
    
    return dA_dp

# --------------------------------------------------        

def p_integral(A, p, logger=None):
    
    if(not isinstance(A, xr.core.dataarray.DataArray)): 
        A = xr.DataArray(A)
    if(not isinstance(p, xr.core.dataarray.DataArray)): 
        p = xr.DataArray(p)
    
    intAdp        = A.copy(deep=True)
    intAdp.values = np.zeros(A.shape)
    for k in range(len(p)):
        intAdp[:,k,:] = np.trapz(A[:,:k+1,:], p[:k+1], axis=1)
    
    if(A.name is not None):
        intAdp.name               = 'int{}dp'.format(A.name)
    if 'long_name' in A.attrs:
        intAdp.attrs['long_name'] = 'vertical integral of {}'.format(A.attrs['long_name'])
    elif A.name is not None:
        intAdp.attrs['long_name'] = 'vertical derivative of {}'.format(A.name)
    if 'units' in A.attrs and 'units' in p.attrs:
        intAdp.attrs['units']     = '{}/{}'.format(A.units, p.units)
    
    return intAdp

# --------------------------------------------------        

def format_latlon_data(data, lat_name='lat', lon_name='lon', 
                       latbnd_name='lat_bnds', lonbnd_name='lon_bnds', 
                       bnddim_name='nbnd'):
    '''
    This function allows users to compute TEM quantities with the PyTEMDiags 
    package on structured lat-lon datasets. This package computes 
    zonal averages via a spherical-harmonic method generalized for unstructred 
    data, and so in order to use it with structured lat-lon data, the data must 
    first be re-formatted in a way that reduces the two lat,lon horizontal 
    dimensions to a single unstructured dimension.
    If the data in the two original structured horizontal 
    dimensions were of length NLAT and NLON, they are stacked in the formatted 
    dataset to form a single dimension of length NCOL = NLAT*NLON named 'ncol'.
    It is assumed that the structured horizontal dimensions of the input data have
    associated coordiantes; those coordiante values will be retianed as variables 
    in the formatted dataset.

    Parameters
    ----------
    data : N-dimensional xarray Dataset.
        The input structured data. Number of dimensions N must be >= 2, with
        the horizontal dimensions specified by the parameters 'lat_name' and 'lon_name'.
        Units of the lat dimension must be degrees, with -90, 90 degrees at the
        poles.
        Units of the lon dimension must be degrees from [0, 360].
    lat_name : str, optional
        Name of the latitude dimension. Defaults to 'lat'.
    lon_name : str, optional
        Name of the longitude dimension. Defaults to 'lon'.
    latbnd_name : str, optional
        Name of latitude bounds variable. Defaults to 'lat_bnds'.
        If the input data does not have this variable, this argument
        does not need to be specified. If the input data does have
        this variable, then latbnd_name, lonbnd_name, and bnddim_name
        all need to be specified.
    lonbnd_name : str, optional
        Name of longitude bounds variable. Defaults to 'lon_bnds'.
        If the input data does not have this variable, this argument
        does not need to be specified. If the input data does have
        this variable, then latbnd_name, lonbnd_name, and bnddim_name
        all need to be specified.
    bnddim_name : str, optional
        Name of longitude, longitude bounds dimension. Defaults to 'nbnd'.
        If the input data has variables with names latbnnd_name and lonbnd_name, 
        then this argument must correctly specify the dimension name used for
        the upper and lower bounds in those variables.

    Returns
    -------
    data : (N-1)-dimensional xarray Dataset.
        The formatted data with an unstructed (1-D) horizontal dimension 'ncol' of 
        length NN
    '''
   
    # get existing lat,lon dimensions
    lat  = data[lat_name]
    lon  = data[lon_name]

    # build grid cell bound vectors if not exist
    # assume that bounds of grid cells lie at midpoint between neighors
    if not (latbnd_name in data.variables):
        latdiff           = np.diff(np.hstack([ lat, lat[-1]+(lat[-1]-lat[-2]) ]))
        lat_bnds          = np.vstack([lat - latdiff/2, lat + latdiff/2]).T
        data[latbnd_name] = (lat.dims + (bnddim_name,), lat_bnds)
    else:
        if bnddim_name not in data[latbnd_name].dims:
            raise RuntimeError('Variable {} does not have dimension {}. Dimensions '\
                               'are: {}. Did you specify the latbnd_name, lonbnd_name, '\
                               'and bnddim_name arguments to format_latlon_data() '\
                               'correctly?'.format(latbnd_name, bnddim_name, 
                                                   data[latbnd_names].dims))
    if not (lonbnd_name in data.variables):
        londiff           = np.diff(np.hstack([ lon, lon[-1]+(lon[-1]-lon[-2]) ]))
        lon_bnds          = np.vstack([lon-londiff/2, lon + londiff/2]).T
        data[lonbnd_name] = (lon.dims+ (bnddim_name,), lon_bnds)
    else:
        if bnddim_name not in data[lonbnd_name].dims:
            raise RuntimeError('Variable {} does not have dimension {}. Dimensions '\
                               'are: {}. Did you specify the latbnd_name, lonbnd_name, '\
                               'and bnddim_name arguments to format_latlon_data() '\
                               'correctly?'.format(lonbnd_name, bnddim_name, 
                                                   data[lonbnd_names].dims))
    # construct 1D ncol dimension
    ncol = np.arange(0, len(lat)*len(lon))
    data = data.stack(ncol=(lat_name, lon_name)).transpose('ncol', ...)

    # save lat,lon coordinates as dataset variables.
    lats = data[lat_name].values
    lons = data[lon_name].values
    
    # remove existing lat,lon coordinates, add back as variables
    data = data.drop_vars((lat_name, lon_name))
    data[lat_name] = ('ncol', lats)
    data[lon_name] = ('ncol', lons)

    return data
