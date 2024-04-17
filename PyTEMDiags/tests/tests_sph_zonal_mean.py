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
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors
import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# ---- imports from this package
from ..tem_diagnostics import *
from ..tem_util import *
from ..sph_zonal_mean import *
from ..constants import *

# ---- global constants 
testsloc    = os.path.dirname(os.path.realpath(__file__))
test_data   = '{}/test_data'.format(testsloc)
test_netcdf = '{}/testdata.nc'.format(test_data)
mapdir      = '{}/../../maps'.format(testsloc)


# -------------------------------------------------------------------------

def test_zonal_mean_latlon_data():
    '''
    This test computes the spherical-harmonic-based zonal mean on a set of *structured*
    lat-lon data. This is done by "raveling" the data, definind an "ncol" dimension, and
    treating the input strucutred data as if it were unstructured. This enables us to 
    compare the results of this package's averaging routine with a arithmetic zonal mean, 
    using the same data. This is a more meaningful comparison than starting with a set of 
    unstructured data, and comparing the spherical-harmonic-based mean to the arithmetic 
    zonal mean of the remapped data.
    '''
   
    datafile = '{}/E3SMv2.limvar.ens1.TEM_VARS_remap_180x360.nc'.format(test_data)
    print('opening {}...'.format(datafile.split('/')[-1]))
    data = xr.open_dataset(datafile)
    lev = data.lev

    # reshape data to unstructured format
    print('reshaping latlon data to mimic unstructured format...')
    lat_out  = data.lat.values
    ncol     = np.arange(0, len(lat_out)*len(data.lon.values))
    data_sph = data.stack(ncol=('lat', 'lon')).transpose('ncol', 'lev')
    lat_sph  = data_sph.lat.values
    data_sph = data_sph.drop_vars(('lat', 'lon'))
    
    print('creating aveaging object...')
    L        = 30
    ZM       = sph_zonal_averager(lat_sph, lat_out, L, 
                                  grid_name='180x360', save_dest=mapdir, debug=True)
    ZM.sph_compute_matrices(no_write=True)
    pdb.set_trace()

    # --- get spherical-harmonic-based mean of unstructured data
    print('getting spherical-harmonic-based zonal means...')
    zm_sph_om = ZM.sph_zonal_mean(data_sph['OMEGA'])
    zm_sph_u  = ZM.sph_zonal_mean(data_sph['U'])
    zm_sph_v  = ZM.sph_zonal_mean(data_sph['V'])
    zm_sph_t  = ZM.sph_zonal_mean(data_sph['T'])
    
    # --- get arithmetic mean of structured data
    print('getting arithmetic zonal means...')
    zm_arith    = data.mean('lon').transpose('lat', 'lev')
    zm_arith_om = zm_arith['OMEGA']
    zm_arith_u  = zm_arith['U']
    zm_arith_v  = zm_arith['V']
    zm_arith_t  = zm_arith['T']

    # --- get relative difference
    zm_diff_u  = np.log10(np.abs(zm_sph_u-zm_arith_u))
    zm_diff_v  = np.log10(np.abs(zm_sph_v-zm_arith_v))
    zm_diff_t  = np.log10(np.abs(zm_sph_t-zm_arith_t))
    zm_diff_om = np.log10(np.abs(zm_sph_om-zm_arith_om))

    # --- make figure
    fig, axes = plt.subplots(4,3, figsize=(12, 15))
    axes = np.ravel(axes)

    # set saturatiomn percentile
    p1 = 2
    p2 = 98

    levels = np.linspace(np.round(np.percentile(zm_arith_u, p1), 0), 
                         np.round(np.percentile(zm_arith_u, p2), 0), 15)
    ucf = axes[0].contourf(lat_out, lev, zm_sph_u.T, levels=levels, cmap='rainbow', extend='both')
    axes[1].contourf(lat_out, lev, zm_arith_u.T, levels=ucf.levels, cmap='rainbow', extend='both')
    ucfd = axes[2].contourf(lat_out, lev, zm_diff_u.T, levels=10, cmap='viridis')
    plt.colorbar(ucf, ax=axes[0], location='top', label='U zm_sph [m/s]')
    plt.colorbar(ucf, ax=axes[1], location='top', label='U zm [m/s]')
    plt.colorbar(ucfd, ax=axes[2], location='top', label='log10(U diff [%])')
 
    levels = np.linspace(np.round(np.percentile(zm_arith_v, p1), 1), 
                         np.round(np.percentile(zm_arith_v, p2), 1), 15)
    vcf = axes[3].contourf(lat_out, lev, zm_sph_v.T, levels=levels, cmap='rainbow', extend='both')
    axes[4].contourf(lat_out, lev, zm_arith_v.T, levels=vcf.levels, cmap='rainbow', extend='both')
    vcfd = axes[5].contourf(lat_out, lev, zm_diff_v.T, levels=10, cmap='viridis')
    plt.colorbar(vcf, ax=axes[3], location='top', label='V zm_sph [m/s]')
    plt.colorbar(vcf, ax=axes[4], location='top', label='V zm [m/s]')
    plt.colorbar(vcfd, ax=axes[5], location='top', label='log10(V diff [%])')
    
    levels = np.linspace(np.round(np.percentile(zm_arith_om, p1), 3), 
                         np.round(np.percentile(zm_arith_om, p2), 3), 15)
    omcf = axes[6].contourf(lat_out, lev, zm_sph_om.T, levels=levels, cmap='rainbow', extend='both')
    axes[7].contourf(lat_out, lev, zm_arith_om.T, levels=omcf.levels, cmap='rainbow', extend='both')
    omcfd = axes[8].contourf(lat_out, lev, zm_diff_om.T, levels=10, cmap='viridis')
    plt.colorbar(omcf, ax=axes[6], location='top', label='OMEGA zm_sph [Pa/s]')
    plt.colorbar(omcf, ax=axes[7], location='top', label='OMEGA zm [Pa/s]')
    plt.colorbar(omcfd, ax=axes[8], location='top', label='log10(OMEGA diff [%])')
    
    levels = np.linspace(np.round(np.percentile(zm_arith_t, p1), 0), 
                         np.round(np.percentile(zm_arith_t, p2), 0), 15)
    tcf = axes[9].contourf(lat_out, lev, zm_sph_t.T, levels=levels, cmap='rainbow', extend='both')
    axes[10].contourf(lat_out, lev, zm_arith_t.T, levels=tcf.levels, cmap='rainbow', extend='both')
    tcfd = axes[11].contourf(lat_out, lev, zm_diff_t.T, levels=10, cmap='viridis')
    plt.colorbar(tcf, ax=axes[9], location='top', label='T [K]')
    plt.colorbar(tcf, ax=axes[10], location='top', label='T [K]')
    plt.colorbar(tcfd, ax=axes[11], location='top', label='log10(T diff [%])')
    
    for ax in axes:
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylabel('lev [hPa]')
        ax.set_xlabel('lat')

    plt.tight_layout()
    plt.savefig('./test_figures/test_zonal_mean_latlon_data_results.png', dpi=200)

    pdb.set_trace()



# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


def test_sph_decomp():
    '''
    The same as test_zonal_mean_latlon_data(), with the modification that we take the
    arithmetic zonal mean of the input structured data first, and pass this zonally-averaged
    data to the spherical-harmonic-based averager. In this way, the tool is not actually
    taking a zonal mean, but is rather simply expanding a pre-computed zonal average into
    a sum of spherical harmonic basis functions. This test is used to test the basic function
    of the computational method being used of solving for the expansion coefficients.
    '''
   
    datafile = '{}/E3SMv2.limvar.ens1.TEM_VARS_remap_180x360.nc'.format(test_data)
    print('opening {}...'.format(datafile.split('/')[-1]))
    data = xr.open_dataset(datafile)
    lev = data.lev
 
    # --- get arithmetic mean of structured data
    print('getting arithmetic zonal means...')
    data  = data.mean('lon').transpose('lat', 'lev')
    data  = data.thin({'lat':2}) # subsample data to 90 points in latitude
    zm_om = data['OMEGA']
    zm_u  = data['U']
    zm_v  = data['V']
    zm_t  = data['T']
    
    # reshape data to unstructured format
    print('reshaping latlon data to mimic unstructured format...')
    lat      = data.lat.values
    ncol     = np.arange(0, len(lat))
    data_sph = data.rename_dims(lat='ncol').rename_vars(lat='ncol').drop_vars('ncol')
    
    print('creating aveaging object...')
    L        = 45
    weights  = np.cos(lat * np.pi/180)
    ZM       = sph_zonal_averager(lat, lat, weights, L, 
                                  grid_name='90x360', save_dest=mapdir, debug=True)
    ZM.sph_compute_matrices(no_write=True)

    # --- get spherical-harmonic-based mean of unstructured data
    print('getting spherical-harmonic-based zonal means...')
    zm_sph_om = ZM.sph_zonal_mean(data_sph['OMEGA'])
    zm_sph_u  = ZM.sph_zonal_mean(data_sph['U'])
    zm_sph_v  = ZM.sph_zonal_mean(data_sph['V'])
    zm_sph_t  = ZM.sph_zonal_mean(data_sph['T'])
        
    # scale omega
    zm_om *= 1e3
    zm_sph_om *= 1e3

    # --- get relative percent difference
    zm_diff_u  = np.abs(zm_sph_u - zm_u)/np.maximum(np.abs(zm_sph_u), np.abs(zm_u)) * 100
    zm_diff_v  = np.abs(zm_sph_v - zm_v)/np.maximum(np.abs(zm_sph_v), np.abs(zm_v)) * 100
    zm_diff_t  = np.abs(zm_sph_t - zm_t)/np.maximum(np.abs(zm_sph_t), np.abs(zm_t)) * 100
    zm_diff_om  = np.abs(zm_sph_om - zm_om)/np.maximum(np.abs(zm_sph_om), np.abs(zm_om)) * 100
    
    do_multipanel = True
    if (do_multipanel):
        # --- make figure
        fig, axes = plt.subplots(4,3, figsize=(12, 15))
        axes = np.ravel(axes)

        # set saturatiomn percentile
        p1v = 10
        p2v = 90
        p1om = 40
        p2om = 60
        diffcmap = plt.cm.viridis
        difflevels = [0, 0.1, 1, 5, 10, 50, 100, 200]
        diffnorm = BoundaryNorm(difflevels, diffcmap.N)

        levels=15
        divnorm=colors.TwoSlopeNorm(vmin=np.min(zm_u), vcenter=0., vmax=np.max(zm_u))
        ucf = axes[0].contourf(lat, lev, zm_sph_u.T, levels=levels, 
                               cmap='rainbow', extend='both', norm=divnorm)
        axes[1].contourf(lat, lev, zm_u.T, levels=ucf.levels, 
                         cmap='rainbow', extend='both', norm=divnorm)
        ucfd = axes[2].contourf(lat, lev, zm_diff_u.T, levels=difflevels, 
                                cmap=diffcmap, extend='both', norm=diffnorm)
        plt.colorbar(ucf, ax=axes[0], location='top', label='U zm_sph [m/s]')
        plt.colorbar(ucf, ax=axes[1], location='top', label='U zm [m/s]')
        plt.colorbar(ucfd, ax=axes[2], location='top', label='log10(U diff [%])')
     
        levels = np.arange(np.round(np.percentile(zm_v, p1v), 1), 
                             np.round(np.percentile(zm_v, p2v), 1)+0.1, 0.1)
        divnorm=colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0., vmax=np.max(levels))
        vcf = axes[3].contourf(lat, lev, zm_sph_v.T, levels=levels,
                               cmap='rainbow', extend='both', norm=divnorm)
        axes[4].contourf(lat, lev, zm_v.T, levels=vcf.levels, 
                         cmap='rainbow', extend='both', norm=divnorm)
        vcfd = axes[5].contourf(lat, lev, zm_diff_v.T, levels=difflevels, 
                                cmap=diffcmap, extend='both', norm=diffnorm)
        plt.colorbar(vcf, ax=axes[3], location='top', label='V zm_sph [m/s]')
        plt.colorbar(vcf, ax=axes[4], location='top', label='V zm [m/s]')
        plt.colorbar(vcfd, ax=axes[5], location='top', label='log10(V diff [%])')
       
        levels = np.arange(np.round(np.percentile(zm_om, p1om), 1), 
                             np.round(np.percentile(zm_om, p2om), 1)+0.02, 0.02)
        divnorm=colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0., vmax=np.max(levels))
        omcf = axes[6].contourf(lat, lev, zm_sph_om.T, levels=levels, 
                                cmap='rainbow', extend='both', norm=divnorm)
        axes[7].contourf(lat, lev, zm_om.T, levels=omcf.levels, 
                         cmap='rainbow', extend='both', norm=divnorm)
        omcfd = axes[8].contourf(lat, lev, zm_diff_om.T, levels=difflevels, 
                                 cmap=diffcmap, extend='both', norm=diffnorm)
        plt.colorbar(omcf, ax=axes[6], location='top', label='OMEGA zm_sph [1e-3 Pa/s]')
        plt.colorbar(omcf, ax=axes[7], location='top', label='OMEGA zm [1e-3 Pa/s]')
        plt.colorbar(omcfd, ax=axes[8], location='top', label='log10(OMEGA diff [%])')
        
        levels=15
        tcf = axes[9].contourf(lat, lev, zm_sph_t.T, levels=levels, 
                               cmap='rainbow', extend='both')
        axes[10].contourf(lat, lev, zm_t.T, levels=tcf.levels, 
                          cmap='rainbow', extend='both')
        tcfd = axes[11].contourf(lat, lev, zm_diff_t.T, levels=difflevels, 
                                 cmap=diffcmap, extend='both', norm=diffnorm)
        plt.colorbar(tcf, ax=axes[9], location='top', label='T [K]')
        plt.colorbar(tcf, ax=axes[10], location='top', label='T [K]')
        plt.colorbar(tcfd, ax=axes[11], location='top', label='log10(T diff [%])')
        
        for ax in axes:
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.set_ylabel('lev [hPa]')
            ax.set_xlabel('lat')

        plt.tight_layout()
        plt.savefig('./test_figures/test_sph_decomp_results.png', dpi=200)
        pdb.set_trace()

    # ----------------------
    # inspect fit on individual levels in temperature

    tt = zm_t.sel(lev=100, method='nearest')
    tts = zm_sph_t.sel(lev=100, method='nearest')
    plt.figure()
    plt.plot(np.deg2rad(lat-90), tt, '.r')
    plt.plot(np.deg2rad(lat-90), tts, '-k')
    plt.xlabel('φ')
    plt.ylabel('T')
    plt.title('L = {}'.format(L))
    plt.savefig('./test_figures/test_sph_decomp_results_100hPa.png'.format(L), dpi=200)

    #pdb.set_trace()



def test_zonal_mean():
   
    dat = xr.open_dataset(test_netcdf).thin({'ncol':2})

    lat       = dat['lat'].values
    lon       = dat['lon'].values
    coalt     = 90 - lat # transform to coaltitude
    lat_out   = np.arange(-89.5, 90.5, 1)
    lon_out   = np.zeros(lat_out.shape)
    coalt_out = 90 - lat_out # transform to coaltitude
    weights   = np.cos(lat * np.pi/180)

    #pow2 = np.arange(11)
    #LL = 2**pow2
    
    LL = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 325, 350, 375, 400, 425, 450]
    
    maxrelerr = np.zeros(len(LL))


    for l in range(len(LL)):

        L = LL[l]

        ZM = sph_zonal_averager(lat, lat_out, weights=weights, L=L, 
                                grid_name='ne30pg2', save_dest=mapdir, debug=True)
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
        
        if(1):
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
            plt.savefig('test_figures/tight_test_L{}.png'.format(L))

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
