# Joe Hollowed
# University of Michigan 2023
#
# Providing a function for generating averaging and remap matrices for
# computing zonal averages of variables on unstructured grids


# =========================================================================


# ---- import dependencies
import pathlib
import warnings
from scipy.linalg import lstsq as lstsq
from scipy.special import eval_legendre as legendre

# ---- imports from this package
import util
from constants import *

# ---- global constants
SAVE_DEST = '{}/../maps'.format(pathlib.Path(__file__).parent.resolve())


# -------------------------------------------------------------------------


def sph_zonal_mean(lat, L, lat_out=None, grid_name=None, grid_out_name=None, 
                   save_dest=None, overwrite=False, debug=False):
    '''
    This function generates a zonal "averaging matrix" using spherical harmonic decomposition
    given latitudes and degree (L). It is also optional to specify a spearate set of output
    latitudes, in which case auxillary "remap matrix" will be returned, which allows interpolation
    of the zonally symmetric spherical harmonic representation to a coarser set of latitudes than
    the native grid.
    The result is written out the a NetCDF file. On subsequent calls to this function given an
    identical configuration, the avergaing and remap matrices will be read from this file, 
    rather than computed (unless the argument overwrite=True).
    This code is based on MATLAB code from Tom Ehrmann at Sandia National Labs.

    Parameters
    ----------
    lat : 1D array
        Unstructured vector of N latitudes of the native data.
    L : int
        Maximum spherical harmonic order.
    lat_out : 1D array, optional
        Secondary output latitudes, in degrees. If not passed, only
        the averaging matrix Z will be built, and not the remap matrix Z'
    grid_name : str, optional
        Name of the native grid. Used for naming the file which the
        averaging and remap matrices will be saved to, as:
        'averaging_matrix_{grid_name}_{grid_out_name}_L{L}.nc'
        If not provided, then grid_name = ncol{ncol}
    grid_out_name : str, optional
        Name of the optional output grid, analogous to grid_name.
        If dlat_out is provided, but this argument is not, then 
            grid_name_out = {dlat}deg
        where dlat is the spacing between the first two elements of lat_out
        (that is, this default naming scheme implies that lat_out is uniform)
        If lat_out is not provided, then this argument will be 
        ignored if passed.
    save_dest : str, optional
        Location at which to save resulting matrices to netcdf file(s).
        Defaults to '../maps' relative to this Python module.
    overwrite : bool, optional
        Whether or not to force re-computation and overwriting, if 
        a filename corresponding to this execution of the function 
        already exists. 
    debug : bool, optional
        Whether or not to print progress statements to stdout.

    Returns
    -------
    Z, or [Z, Zp] : NxN array, or list of [NxN, MxM] arrays
        Z:  The resulting square (NxN) zonal averaging matrix.
        Zp: The resulting non-square (MxN) zonal averaging remap matrix.
    '''

    logger = util.logger(debug, 'SPH_ZONAL_MEAN')
   
    # ---- check input args
    if(lat_out is None and grid_out_name is not None):
        warnings.warn('dlat_out was unpassed, but grid_out_name was passed; ignoring grid_out_name')
     
    # ---- get data sizes, degree array
    N = len(lat)               # number of input data points
    l  = np.arange(L+1)        # spherical harmonic degree
    if(lat_out is not None):
        M = len(lat_out)           # number of output data points
        logger.print('calling sph_zonal_mean for (M x N) = ({} x {}), L = {}'.format(M, N, L))
    else:
        logger.print('calling sph_zonal_mean for N = {}, L = {}'.format(N, L))
    
    # ---- identify output files
    if(save_dest is None):
        save_dest = SAVE_DEST
    if(grid_name is None):
        grid_out = 'ncol{}'.format(N)
    Z_file_out = '{}/Z_{}_L{}.nc'.format(save_dest, grid_out, L)
    if(dlat_out is None):
        Zp_file_out = None
    else:
        if(grid_out_name is None):
            grid_out_name = '{}deg'.format(dlat_out)
        Z_file_out = '{}/Zp_{}_{}_L{}.nc'.format(save_dest, grid_out, grid_out_name, L)
    
    # ---- read the averaging and remap matrices from file, if exists
    try:
        if(overwrite): raise FileNotFoundError
        Z  = xr.open_dataset(Z_file_out)
        Zp = xr.open_dataset(Zp_file_out)
        logger.print('Z read from file {}'.format(Z_file_out))
        if(lat_out is not None):
            logger.print('Z\' read from file {}'.format(Zp_file_out))
            return [Z, Zp]
        else: return Z
    except FileNotFoundError:
        pass
 
    # ---- compute zeroth-order spherical harmonics Y[l,m=0] at the input lats
    Y0 = np.zeros((N, L+1))    # matrix to store spherical harmonics on input lats
    sinlat = np.sin(np.deg2rad(lat))
    logger.print('building Y0...')
    for ll in l:
        coef = np.sqrt(((2*ll+1)/(4*pi)))
        Y0[:,ll] = coef * legendre(ll, sinlat)

    # ---- if desired, compute zeroth-order spherical harmonics Y[l,m=0] at the output lats
    if(lat_out is not None):
        M   = len(lat_out)         # number of output data points
        Y0p = np.zeros((M, L+1))   # matrix to store spherical harmonics on output lats
        sinlat = np.sin(np.deg2rad(lat_out))
        logger.print('building Y0\'...')
        for ll in l:
            coef = np.sqrt(((2*ll+1)/(4*pi)))
            Y0p[:,ll] = coef * legendre(ll, sinlat)

    # ---- construct the remap matrices
    #
    # ZM recovers the zonal mean on a reduced grid with M data points
    # ZM_nat recovers the zonal mean on the native grid with N data points
    #
    # These lines are the bulk of the computation of this function; the matrix inversions and 
    # subsequent multplications are timed, and the results printed, if debug=True.
    # After solving, the matrices Z,Zp are written out to file. The format is NetCDF; for 
    # completeness, we name the variable, dimensions, and provide a long name.
    
    logger.print('inverting Y0...', with_timer=True)
    Y0inv = lstsq(Y0, np.identity(N))[0]
    logger.timer()
    
    logger.print('taking Z = Y0*inv(Y0)...', with_timer=True)
    Z     = np.matmul(Y0, Y0inv)
    logger.timer()
    
    logger.print('writing Z to file {}'.format(Z_file_out))
    Z_da  = xr.DataArray(Z, dims=('lat_row','lat_col'),
                            coords={'lat': lat, 'lat':lat})
    Z_da.name = 'Z'
    Z_da.attrs['long_name'] = 'Averaging matrix Z for grid {}'.format(grid_name)
    Z_da.to_netcdf(Z_file_out)

    if(lat_out is None): return Z

    else:
        logger.print('taking Z\' = Y0\'*inv(Y0)...', with_timer=True)
        Zp = np.matmul(Y1, Y0inv)
        logger.timer()
        
        logger.print('writing Z\' to file {}'.format(Z_file_out))
        Zp_da  = xr.DataArray(Zp, dims=('lat_row','lat_col'),
                                 coords={'lat': lat, 'lat':lat})
        Zp_da.name = 'Zp'
        Zp_da.attrs['long_name'] = 'Remap matrix Z\' for grid {}'.format(grid_name)
        Zp_da.to_netcdf(Z_file_out)
   
        return [Z, Zp]


# -------------------------------------------------------------------------

