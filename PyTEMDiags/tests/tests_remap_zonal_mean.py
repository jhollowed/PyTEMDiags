# Joe Hollowed
# University of Michigan 2023
#
# Providing a suite of tests to verify the methods of the sph_zonal_averager class


# =========================================================================


# ---- import dependencides
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from nose.tools import set_trace
from scipy.special import sph_harm
from matplotlib.ticker import ScalarFormatter as scalarfmt
import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# ---- imports from this package
from ..tem_diagnostics import *
from ..tem_util import *
from ..sph_zonal_mean import *
from ..constants import *

# ---- global constants 
testsloc = os.path.dirname(os.path.realpath(__file__))
test_netcdf = '{}/data/testdata.nc'.format(testsloc)
mapdir = '{}/../maps'.format(testsloc)


# -------------------------------------------------------------------------


def test_zonal_mean_remapped():
   
    datf2 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/f2.regrid.180x360_aave.nc')
    #datf2 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/f2.regrid.180x360_bilinear.nc')
    datY20 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/Y20.regrid.180x360_aave.nc')
    #datY20 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/Y20.regrid.180x360_bilinear.nc')

    print('-------- REMAP TEST')

    lat       = datf2['lat'].values
    lon       = datf2['lon'].values
    LON, LAT = np.meshgrid(lon, lat)
    LON, LAT = np.ravel(LON), np.ravel(LAT)
    COALT     = 90 - LAT # transform to coaltitude
    
    lat_out   = np.arange(-89.5, 90.5, 1)
    lon_out   = np.zeros(lat_out.shape)

    f2zm = datf2['f2'].mean('lon')
    Y20zm = datY20['Y20'].mean('lon')

    test_func_Y20 = lambda lon, coalt: sph_harm(0, 2, np.deg2rad(lon), np.deg2rad(coalt))
    test_func_f2  = lambda lon, lat: np.deg2rad(lat) ** 2 + 1 
    Y20_verify    = test_func_Y20(0, lat).reshape(Y20zm.shape)
    f2_verify     = test_func_f2(0, lat).reshape(f2zm.shape)

    maxrelerr = np.max(np.hstack(np.abs((f2zm-f2_verify)/f2_verify) * 100))
    print('------MAXERR: {}'.format(maxrelerr))
    
    np.save('maxerr_remap_aave.npy', maxrelerr)
   
    return 
