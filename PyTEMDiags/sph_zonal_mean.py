# Joe Hollowed
# University of Michigan 2023
#
# Providing a function for generating averaging and remap matrices for
# computing zonal averages of variables on unstructured grids


# =========================================================================


# ---- import dependencies
import os
import pdb
import pathlib
import warnings
import numpy as np
import xarray as xr
from scipy.linalg import lstsq as lstsq
from scipy.special import eval_legendre as legendre

# ---- imports from this package
from .tem_util import *
from .constants import *

# ---- global constants
SAVE_DEST = '{}/../maps'.format(pathlib.Path(__file__).parent.resolve())
DEFAULT_LAT_ATTRS = {'long_name':'Latitude of Grid Cell Centers', 'standard_name':'latitude', 
                     'units':'degrees_north', 'axis':'Y'} 
NCENC = {'_FillValue':None}


# -------------------------------------------------------------------------


class sph_zonal_averager:
    def __init__(self, lat, lat_out, L, grid_name=None, grid_out_name=None, save_dest=None,
                 ncoldim='ncol', debug=False):
        '''
        This class provides an interface for taking zonal averages of fields provided
        on unstructured datasets on the globe using a spherical harmonic decomposition 
        method. The input latitudes 'lat' will be called the "native grid", while the
        output latitudes 'lat_out' will be called the "output grid".
        After creation of the object, the averaging and remap matrices must be computed
        (or read if already exist) by calling sph_zm_matrices()
        Then, variables may be zonally averaged by calling either sph_zonal_mean(), or
        sph_zonal_mean_native().
        This method is designed to provide two different expressions of the zonal mean,
        one that returns a zonal mean for *every* input datapoint on the native grid, 
        and one that returns a zonal mean for only the output grid. This allows
        interpolation of the zonally symmetric spherical harmonic representation to a 
        coarser set of latitudes than the native grid.
        This code is based on MATLAB code from Tom Ehrmann at Sandia National Labs.

        Parameters
        ----------
        lat : 1D array
            Unstructured vector of N latitudes of the native data.
        lat_out : 1D array
            Output latitudes, in degrees.
        L : int
            Maximum spherical harmonic order.
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
            Defaults to '../maps' relative to this Python module. Note that 
            the maps saved here will be very large! If this clone of the PyTEMDiags
            repo is sitting in a location with limited storage, it is highly
            reccomended to supply this argument.
        ncoldim : str, optional
            name of horizontal dimension. Defaults to 'ncol'
        debug : bool, optional
            Whether or not to print progress statements to stdout.

        Public Methods
        --------------
        sph_zonal_mean()
            Takes the zonal mean of an input variable on the native grid, returning the 
            zonal mean on the output grid.
        sph_zonal_mean_native()
            Takes the zonal mean of an input variable on the native grid, returning the 
            zonal mean on the native grid.
        compute_zm_matrices()
            Generates the zonal "averaging matrix" and "remap matrix"  using spherical harmonic 
            decomposition, given latitudes and degree (L).

        Public Attributes
        -----------------
        All input parameters are retained as class attrbiutes with the same name. In addition:
        N : int
            Number of input native grid latitudes.
        M : int 
            Number of output gtrid latitudes.
        Z : (N x N) array
            Averaging matrix for estiamting zonal means of the native data on the native grid
        Zp : (M x N) array
            Remap matrix for estiamting zonal means of the native data on the output grid
        Z_file_out : str
            NetCDF file location for writing/reading matrix Z.
        Zp_file_out : str
            NetCDF file location for writing/reading matrix Zp.
        '''
        
        self.logger = logger(debug, 'sph_zonal_mean')

        # ---- arguments
        self.L = L                         # max. spherical harmonic order  
        self.lat = lat                     # input latitudes [deg]
        self.lat_out = lat_out             # output latitudes [deg]
        self.grid_name = grid_name         # name of native (input) grid
        self.grid_out_name = grid_out_name # name of optional output grid
        self.save_dest = save_dest         # save location for matric netcdf files
        self.ncoldim = ncoldim             # name of horizontal dimension for input data

        # if lat was passed as a DataArray, extract values
        if(isinstance(self.lat, xr.core.dataarray.DataArray)): 
            self.lat = self.lat.values
        
        # ---- declare variables
        self.N = len(lat)       # number of input latitudes
        self.M = len(lat_out)   # number of optional output latitudes
        self.l = np.arange(L+1) # spherical harmonic degree 0->L
        self.Z = None           # zonal averageing matrix (native -> native)
        self.Zp = None          # zonal remap matrix (native -> output)
        self.Z_file_out = None  # output file location for matrix Z
        self.Zp_file_out = None # output file location for matrix Zp
         
        # ---- identify matrix output files, apply defaults 
        if(save_dest is None):
            save_dest = SAVE_DEST
        if(grid_name is None):
            grid_name = 'ncol{}'.format(self.N)
        self.Z_file_out = '{}/Z_{}_L{}.nc'.format(save_dest, grid_name, L)
        if(grid_out_name is None):
            dlat_out = np.diff(self.lat_out)[0]
            grid_out_name = '{}deg'.format(dlat_out)
        self.Zp_file_out = '{}/Zp_{}_{}_L{}.nc'.format(save_dest, grid_name, grid_out_name, L)

        # ---- read remap matrices, if currently exist on file
        self.sph_compute_matrices(read_only=True)
 

    # --------------------------------------------------
        
         
    def _sph_zonal_mean_generic(self, A, ZZ):
        '''
        Takes the zonal mean of an input native-grid variable A, given the matrix ZZ.

        Parameters
        ----------
        A : length-N 1D xarray DataArray, or N-dimensional xarray DataArray
            An input dataset defined on the N native gridpoints. A single horizontal 
            dataset can be passed as 1D array of length N, or a varying horizontal 
            dataset can be passed as an N-dimensional array. It is assumed and enforced
            that the horizontal dimension of length N is the leftmost dimension of 
            the array. It is also assumed that this horizontal dimension has no associated
            coordinate values, and no attributes.
        ZZ : (NN x N) array
            2D transformation matrix. M must be <= N. If desiring to obtain the zonal
            mean of A on the native grid, then NN==N (i.e. this argument should be self.Z).
            If instead desiring to obtain the zonal mean of A on the output grid, 
            then NN==M<N (i.e. this argument should be self.Zp).

        Returns
        -------
        Abar : length-N 1D xarray DataArray, or N-dimensional xarray DataArray 
            The zonal mean of the input dataset. If ZZ=Z, then the shape of the output
            data Abar matches the input data A. If ZZ=Zp, then the shape of the output
            data Abar matches the input data A in all dimensions except for the 
            horizontal (leftmost) dimension, which is reduced in length from N to M.
            If ZZ=Zp, then the latitudes of the output grid are added to the meta
            data of the DataArray Abar as a coordinate, and in either case for ZZ, 
            the variabel long_name attribute is updated.
        '''

        if(self.Z is None or self.Zp is None):
            raise RuntimeError('Matrices Z and/or Zp are undefined; either verify grid_name,'\
                               'grid_name_out, and save_dest, or call sph_compute_matrices()'\
                               'before sph_zonal_mean() or sph_zonal_mean_native()!')

        # ---- check dimensions of input data
        dims = A.dims
        shape = A.shape
        if(dims[0] != self.ncoldim or shape[0] != self.N):
            raise RuntimeError('(sph_zonal_mean_generic() Expected the first (leftmost) '\
                               'dimension of variable {} to be {} of length {}'.format(
                                                          A.name, self.ncoldim, self.N))
        # ---- extract and reshape data for averaging
        AA = A.values
        DD = np.prod(shape[1:])  # outer dimension of data
        AA = AA.reshape((self.N, DD))
        self.logger.print('Reshaped variable {}: {} -> {}'.format(A.name, shape, AA.shape))
        
        # ---- do averaging, reshape
        self.logger.print('Taking zonal average of variable {}...'.format(A.name))
        Abar = np.matmul(ZZ, AA)
        
        oldshape = Abar.shape
        NN = ZZ.shape[0]         # outer dimension of ZZ (either N or M)
        Abar = Abar.reshape((NN, *shape[1:]))
        self.logger.print('Reshaped zonal mean of variable {}: {} -> {}'.format(
                                                   A.name, oldshape, Abar.shape))

        # ---- convert back to xarray DataArray, return
        # at this point either NN=N, or NN=M. If NN=N, then the input DataArray
        # already has the correct dimensions, and we need only reassign the values 
        # and update the veriable long name. If, on the other hand, NN=M, then the 
        # horizontal dimension needs to be trimmed in the DataArray. Additionally,
        # if NN=M, then we'll assign lat_out to a coordinate in the DataArray 
        # (whereas it is conventional for native grid data to have an integer-valued
        # column ncol)
        if(NN == self.M): 
            A = A.isel(ncol=slice(self.M))
            A = A.rename({'ncol':'lat'})
            A.coords['lat'] = self.lat_out
            A.attrs = DEFAULT_LAT_ATTRS
            self.logger.print('Reduced ncol of variable {} to the zonal-mean grid: '\
                              'ncol={} -> ncol{}'.format(A.name, self.N, self.M))
        A.values = Abar
        A.attrs['long_name'] = 'zonal mean of {}'.format(A.name)
        return A 
         
    def sph_zonal_mean_native(self, A):
        '''
        Calls sph_zonal_mean_generic with ZZ=Z (the native grid). See _sph_zonal_mean_generic
        docstrings for argument descriptions
        '''
        return self._sph_zonal_mean_generic(A, self.Z)
    def sph_zonal_mean(self, A):
        '''
        Calls sph_zonal_mean_generic with ZZ=Zp (the output grid). See _sph_zonal_mean_generic
        docstrings for argument descriptions
        '''
        return self._sph_zonal_mean_generic(A, self.Zp)
    

    # --------------------------------------------------


    def sph_compute_matrices(self, overwrite=False, read_only=False):
        '''
        Generates the zonal "averaging matrix" and "remap matrix"  using spherical harmonic 
        decomposition, given latitudes and degree (L).
        The result is written out the a NetCDF file. On subsequent calls to this function 
        given an identical configuration, the avergaing and remap matrices will be read from 
        this file, rather than computed (unless the argument overwrite=True).
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether or not to force re-computation and overwriting, if 
            a filename corresponding to this execution of the function 
            already exists. 
        read_only : bool, optional
            If True, only allow this function to load in the matrices from
            file. If the files don't exist, return rather than computing 
            them. Defaults to False.

        Returns
        -------
        Z, or [Z, Zp] : NxN array, or list of [NxN, MxM] arrays
            Z:  The resulting square (NxN) zonal averaging matrix.
            Zp: The resulting non-square (MxN) zonal averaging remap matrix.
        '''
 
        # ---- read the averaging and remap matrices from file, if exists
        try:
            if(overwrite):
                if os.path.isfile(self.Z_file_out): os.remove(self.Z_file_out)
                if os.path.isfile(self.Zp_file_out): os.remove(self.Zp_file_out)
            self.Z  = xr.open_dataset(self.Z_file_out)['Z'].values
            self.Zp = xr.open_dataset(self.Zp_file_out)['Zp'].values
            self.logger.print('Z read from file {}'.format(self.Z_file_out))
            self.logger.print('Z\' read from file {}'.format(self.Zp_file_out))
            return
        except FileNotFoundError:
            if(read_only):
                return
            
        
        self.logger.print('called sph_compute_matrices() for (M x N) '\
                          '= ({} x {}), L = {}'.format(self.M, self.N, self.L))
     
        # ---- compute zeroth-order spherical harmonics Y[l,m=0] at the input lats
        Y0 = np.zeros((self.N, self.L+1))    # matrix to store spherical harmonics on input lats
        sinlat = np.sin(np.deg2rad(self.lat))
        self.logger.print('building Y0...')
        for ll in self.l:
            coef = np.sqrt(((2*ll+1)/(4*pi)))
            Y0[:,ll] = coef * legendre(ll, sinlat)

        # ---- compute zeroth-order spherical harmonics Y[l,m=0] at the output lats
        Y0p = np.zeros((self.M, self.L+1))   # matrix to store spherical harmonics on output lats
        sinlat = np.sin(np.deg2rad(self.lat_out))
        self.logger.print('building Y0\'...')
        for ll in self.l:
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
       
        # -- build Z
        self.logger.print('inverting Y0...', with_timer=True)
        Y0inv = lstsq(Y0, np.identity(self.N))[0]
        self.logger.timer()
        
        self.logger.print('taking Z = Y0*inv(Y0)...', with_timer=True)
        Z     = np.matmul(Y0, Y0inv)
        self.logger.timer()
        
        # -- write out Z 
        Z_da  = xr.DataArray(Z, dims=('lat_row','lat_col'),
                                coords={'lat_row': self.lat, 
                                        'lat_col':self.lat})
        Z_da.name = 'Z'
        Z_da.attrs['long_name'] = 'Averaging matrix Z for grid {}'.format(self.grid_name)
        Z_da['lat_row'].attrs['units'] = 'degrees'
        Z_da['lat_col'].attrs['units'] = 'degrees'
        Z_da.to_netcdf(self.Z_file_out, encoding={'Z':NCENC,'lat_row':NCENC,'lat_col':NCENC})
        self.logger.print('Z wrote to file {}'.format(self.Z_file_out))

        # -- build Z'
        self.logger.print('taking Z\' = Y0\'*inv(Y0)...', with_timer=True)
        Zp = np.matmul(Y0p, Y0inv)
        self.logger.timer()
        
        # -- write out Z'
        Zp_da  = xr.DataArray(Zp, dims=('lat_row','lat_col'),
                                 coords={'lat_row': self.lat_out, 
                                         'lat_col':self.lat})
        Zp_da.name = 'Zp'
        Zp_da.attrs['long_name'] = 'Remap matrix Z\' for grid {}'.format(self.grid_name)
        Zp_da['lat_row'].attrs['units'] = 'degrees'
        Zp_da['lat_col'].attrs['units'] = 'degrees'
        Zp_da.to_netcdf(self.Zp_file_out, encoding={'Zp':NCENC,'lat_row':NCENC,'lat_col':NCENC})

        self.logger.print('Zp wrote to file {}'.format(self.Zp_file_out))
        
        # -- done; export to class namespace and return
        self.Z = Z
        self.Zp = Zp
        return


# -------------------------------------------------------------------------

