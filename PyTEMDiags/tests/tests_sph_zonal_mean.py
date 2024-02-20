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


def test_zonal_mean():
   
    dat = xr.open_dataset(test_netcdf)

    lat       = dat['lat'].values
    lon       = dat['lon'].values
    coalt     = 90 - lat # transform to coaltitude
    lat_out   = np.arange(-89.5, 90.5, 1)
    lon_out   = np.zeros(lat_out.shape)
    coalt_out = 90 - lat_out # transform to coaltitude

    #pow2 = np.arange(11)
    #LL = 2**pow2
    
    LL = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 325, 350, 375, 400, 425, 450]
    
    maxrelerr = np.zeros(len(LL))


    for l in range(len(LL)):

        L = LL[l]

        ZM = sph_zonal_averager(lat, lat_out, L=L, grid_name='ne30pg2', save_dest=mapdir, debug=True)
        ZM.sph_compute_matrices(no_write=True)
       
        X   = dat['X'].mean('lev')
        Y20 = X.copy(deep=True)
        Y21 = X.copy(deep=True)
        f1  = X.copy(deep=True)
        f2  = X.copy(deep=True)
     
        test_func_Y21 = lambda lon, coalt: sph_harm(1, 2, np.deg2rad(lon), np.deg2rad(coalt))
        test_func_f1  = lambda lon, coalt: np.sin(np.deg2rad(lon)) 
        Y21.values    = test_func_Y21(lon, coalt)
        f1.values     = test_func_f1(lon, coalt)
        Y21zm         = ZM.sph_zonal_mean(Y21)
        f1zm          = ZM.sph_zonal_mean(f1)
        Y21_verify    = Y21zm * 0
        f1_verify     = f1zm * 0

        test_func_Y20 = lambda lon, coalt: sph_harm(0, 2, np.deg2rad(lon), np.deg2rad(coalt)).real
        test_func_f2  = lambda lon, lat: np.deg2rad(lat) ** 2 + 1 
        Y20.values    = test_func_Y20(lon, coalt)
        f2.values     = test_func_f2(lon, lat)
        Y20zm         = ZM.sph_zonal_mean(Y20)
        f2zm          = ZM.sph_zonal_mean(f2) 
        Y20_verify    = test_func_Y20(lon_out, coalt_out)
        f2_verify     = test_func_f2(lon_out, lat_out)

        # write out data for remap testing...
        if(0):
            dat['X'] = f2
            dat['X'].name='f2'
            dat = dat.rename_vars({'X':'f2'})
            dat.to_netcdf('{}/data/f2.nc'.format(testsloc), encoding={'f2':{'_FillValue':None}})
            dat['f2'] = Y20
            dat['f2'].name='Y20'
            dat = dat.rename_vars({'f2':'Y20'})
            dat.to_netcdf('{}/data/Y20.nc'.format(testsloc), encoding={'Y20':{'_FillValue':None}})
            print('------ SAVED')
        
        if(0):
            fig = plt.figure(figsize=(16, 12))
           
            ax = fig.add_subplot(4, 3, 1)
            ax.tricontourf(lon, lat, Y21.real)
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('lon')
            ax.set_title('Y(l=2,m=1)')
            ax.set_ylim([-95, 95])
            ax = fig.add_subplot(4, 3, 2)
            ax.plot(np.abs(Y21zm.real), lat_out)
            #ax.plot(Y21_verify.real, lat_out, label='truth')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('Y(l=2,m=1) absolute error')
            ax.set_ylim([-95, 95])
            ax.set_xlim([1e-16, 1e15])
            ax.set_xscale('log')
            #ax.legend()
            ax = fig.add_subplot(4, 3, 3)
            ax.set_axis_off()
            
            ax = fig.add_subplot(4, 3, 4)
            ax.tricontourf(lon, lat, f1)
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('lon')
            ax.set_title('f1')
            ax.set_ylim([-95, 95])
            ax = fig.add_subplot(4, 3, 5)
            ax.plot(np.abs(f1zm), lat_out)
            #ax.plot(f1_verify, lat_out, label='truth')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('f1 absolute error')
            ax.set_ylim([-95, 95])
            ax.set_xlim([1e-16, 1e15])
            ax.set_xscale('log')
            #ax.legend()
            ax = fig.add_subplot(4, 3, 6)
            ax.set_axis_off()
            
            ax = fig.add_subplot(4, 3, 7)
            ax.tricontourf(lon, lat, Y20)
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('lon')
            ax.set_title('Y(l=2,m=0)')
            ax.set_ylim([-95, 95])
            ax = fig.add_subplot(4, 3, 8)
            ax.plot(Y20zm, lat_out, label='estimate')
            ax.plot(Y20_verify, lat_out, label='truth')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('Y(l=2,m=0)')
            ax.set_ylim([-95, 95])
            ax.set_xlim([-4, 4])
            ax.legend()
            ax = fig.add_subplot(4, 3, 9)
            ax.plot(np.abs((Y20zm-Y20_verify)/Y20_verify) * 100, lat_out, 'r')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('relative error [%]')
            ax.set_ylim([-95, 95])
            ax.set_xlim([1e-16, 1e3])
            ax.set_xscale('log')
            
            ax = fig.add_subplot(4, 3, 10)
            ax.tricontourf(lon, lat, f2)
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('lon')
            ax.set_title('f2')
            ax.set_ylim([-95, 95])
            ax = fig.add_subplot(4, 3, 11)
            ax.plot(f2zm, lat_out, label='estimate')
            ax.plot(f2_verify, lat_out, label='truth')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('f2')
            ax.set_ylim([-95, 95])
            ax.set_xlim([-8, 10])
            ax.legend()
            ax = fig.add_subplot(4, 3, 12)
            ax.plot(np.abs((f2zm-f2_verify)/f2_verify) * 100, lat_out, 'r')
            ax.grid()
            ax.set_ylabel('lat')
            ax.set_xlabel('relative error [%]')
            ax.set_yscale
            ax.set_ylim([-95, 95])
            ax.set_xlim([1e-16, 1e3])
            ax.set_xscale('log')

            fig.suptitle('L = {}'.format(L))

            plt.tight_layout()
            plt.savefig('tight_test_L{}.png'.format(L))

        maxrelerr[l] = np.max(np.hstack(np.abs((f2zm-f2_verify)/f2_verify) * 100))
    
    np.save('maxerr.npy', maxrelerr)
    #set_trace()
    
    assert np.all(np.isclose(Y21zm, Y21_verify)),\
    'sph_zonal_mean returned nonzero zonal average for Y21!'
    
    assert np.all(np.isclose(Y21zm, Y21_verify)),\
    'sph_zonal_mean returned nonzero zonal average for f1!'

    assert np.all(np.isclose(Y20zm, Y20_verify)),\
    'sph_zonal_mean returned incorrect zonal average for Y20!'
    
    assert np.all(np.isclose(f2zm, f2_verify)),\
    'sph_zonal_mean returned incorrect zonal average for f2!'

    return


def test_zonal_mean_remapped():
   
    #datf2 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/f2.regrid.180x360_aave.nc')
    datf2 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/f2.regrid.180x360_bilinear.nc')
    #datY20 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/Y20.regrid.180x360_aave.nc')
    datY20 = xr.open_dataset('/global/homes/j/jhollo/repos/PyTEMDiags/PyTEMDiags/tests/data/Y20.regrid.180x360_bilinear.nc')

    lat       = dat['lat'].values
    lon       = dat['lon'].values
    coalt     = 90 - lat # transform to coaltitude
    lat_out   = np.arange(-89.5, 90.5, 1)
    lon_out   = np.zeros(lat_out.shape)
    coalt     = 90 - lat # transform to coaltitude

    f2zm = datf2['f2'].mean('lon')
    Y20zm = datY20['f2'].mean('lon')

    test_func_Y20 = lambda lon, coalt: sph_harm(0, 2, np.deg2rad(lon), np.deg2rad(coalt))
    test_func_f2  = lambda lon, lat: np.deg2rad(lat) ** 2 + 1 
    Y20_verify    = test_func_Y20(lon_out, coalt_out)
    f2_verify     = test_func_f2(lon_out, lat_out)

    maxrelerr[l] = np.max(np.hstack(np.abs((f2zm-f2_verify)/f2_verify) * 100))
    
    np.save('maxerr.npy', maxrelerr)
   
    return 
