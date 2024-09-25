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
from scipy.special import sph_harm
from scipy.linalg import lstsq as lstsq

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
    def __init__(self, lat, lat_out, L, weights=None, grid_name=None, grid_out_name=None, 
                 ncoldim='ncol', overwrite=False, save_dest=None, debug=False):
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
            Unstructured vector of N latitudes of the native data, in degrees.
        lat_out : 1D array
            Output latitudes, in degrees.
        L : int
            Maximum spherical harmonic order.
        weights : 1D array or string, optional
            Unstructured vector of N grid area weights for the grid positions lat. These 
            weights must sum to one (each weight is a fractional surface area of the unit 
            sphere).
            Defaults to None, in which case the weights are solved for by a least-squares
            based matrix inversion. Providing these weights will allow a much faster
            and memory-efficient calculation by avoiding the need for this matrix inversion. 
            If weights are not available to be passed, the result of the  inversion method 
            will be written out to file so that subsequent usage of this utility with the 
            same pair of input and output grids will be fast.
            **DEPRECATED; LSTSQ SOLVER IS GOOD ENOUGH**
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
        ncoldim : str, optional
            name of horizontal dimension. Defaults to 'ncol'
        overwrite : bool, optional
            Whether or not to force re-computation and overwriting of the Y0, Yinv and Y0p
            matrices on file, if a filename corresponding to this execution of the 
            function already exists. 
        save_dest : str, optional
            Location at which to save resulting matrices Y0, Y' to netcdf file(s).
            Defaults to '../maps' relative to this Python module. Note that 
            the maps saved here will be very large! If this clone of the PyTEMDiags
            repo is sitting in a location with limited storage, it is highly
            reccomended to supply this argument.
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
        Y0 : (N x L+1) array
            Averaging matrix for estiamting zonal means of the native data on the native grid
        Y0inv : (N x L+1) array
            Inverse of Y0
        Y0p : (M x L+1) array
            Remap matrix for estiamting zonal means of the native data on the output grid
        Y0_file_out : str
            NetCDF file location for writing/reading matrices Y0, Y0inv.
        Y0p_file_out : str
            NetCDF file location for writing/reading matrix Y0p.
        '''
        
        self.logger = logger(debug, 'sph_zonal_mean')

        # ---- arguments
        self.L = L                         # max. spherical harmonic order  
        self.lat = lat                     # input latitudes [deg]
        self.lat_out = lat_out             # output latitudes [deg]
        self.weights = weights             # grid area weights
        self.grid_name = grid_name         # name of native (input) grid
        self.grid_out_name = grid_out_name # name of optional output grid
        self.save_dest = save_dest         # save location for matric netcdf files
        self.ncoldim = ncoldim             # name of horizontal dimension for input data

        # if lats were passed as a DataArray, extract values
        if(isinstance(self.lat, xr.core.dataarray.DataArray)): 
            self.lat = self.lat.values
        if(isinstance(self.lat_out, xr.core.dataarray.DataArray)): 
            self.lat_out = self.lat_out.values
        
        # ---- declare variables
        self.N = len(lat)       # number of input latitudes
        self.M = len(lat_out)   # number of output latitudes
        self.l = np.arange(L+1) # spherical harmonic degree 0->L
        self.diagw = None       # diagonal weight matrix
        self.Y0    = None       # spherical harmonic matrix (native -> native)
        self.Y0inv = None       # inverted spherical harmonic matrix (native -> native)
        self.Y0p    = None       # spherical harmonic remap matrix (native -> output)
        self.Y0_file_out = None  # output file location for matrices Y0, Y0inv
        self.Y0p_file_out = None # output file location for matrix Y0p
         
        # ---- identify matrix output files, apply defaults 
        if(self.save_dest is None):
            self.save_dest = SAVE_DEST
        if(self.grid_name is None):
            self.grid_name = 'ncol{}'.format(self.N)
        self.Y0_file_out = '{}/Y0_{}_L{}.nc'.format(self.save_dest, self.grid_name, self.L)
        if(self.grid_out_name is None):
            dlat_out = np.diff(self.lat_out)[0]
            self.grid_out_name = '{}deg'.format(dlat_out)
        self.Y0p_file_out = '{}/Y0p_{}_{}_L{}.nc'.format(self.save_dest, self.grid_name, 
                                                       self.grid_out_name, self.L)

        # ---- read remap matrices, if currently exist on file
        self.sph_compute_matrices(read_only=True, overwrite=overwrite)

        # --- scale grid weights to unit sphere surface area
        if(self.weights is not None):
            self.weights *= 4*np.pi
                
                    
    # --------------------------------------------------
        
         
    def _sph_zonal_mean_generic(self, A, Y):
        '''
        Takes the zonal mean of an input native-grid variable A, given the spherical harmonic matrix Y.

        Parameters
        ----------
        A : length-N 1D xarray DataArray, or N-dimensional xarray DataArray
            An input dataset defined on the N native gridpoints. A single horizontal 
            dataset can be passed as 1D array of length N, or a varying horizontal 
            dataset can be passed as an N-dimensional array. It is assumed and enforced
            that the horizontal dimension of length N is the leftmost dimension of 
            the array. It is also assumed that this horizontal dimension has no associated
            coordinate values, and no attributes.

        Returns
        -------
        Abar : length-N 1D xarray DataArray, or N-dimensional xarray DataArray 
            The zonal mean of the input dataset. If Y=Y0, then the shape of the output
            data Abar matches the input data A. If Y=Y0p, then the shape of the output
            data Abar matches the input data A in all dimensions except for the 
            horizontal (leftmost) dimension, which is reduced in length from N to M.
            If Y=Y0p, then the latitudes of the output grid are added to the meta
            data of the DataArray Abar as a coordinate, and in either case for Y, 
            the variabel long_name attribute is updated.
        '''

        if(self.Y0 is None or self.Y0p is None):
            raise RuntimeError('Matrices Y0, Y0inv, and/or Y0p are undefined; either verify grid_name,'\
                               'grid_name_out, and save_dest, or call sph_compute_matrices()'\
                               'before sph_zonal_mean() or sph_zonal_mean_native()!')
        if(not isinstance(A, xr.core.dataarray.DataArray)):
            raise RuntimeError('Variable {} must be an xarray DataArray!'.format(A.name))
        if(np.sum(np.isnan(A)) > 0):
            raise RuntimeError('Variable {} has nans! Spectral zonal averager cannot handle nans; '\
                               'please replace or remove them'.format(A.name))

        # ---- A will have been passed by reference; 
        #      make copy of the data object for modification
        A = A.copy(deep=True)
        
        # ---- if variable has no name, assign temporary name (which is more clear than 'None')
        if(A.name is None):
            A.name = '{unnamed variable}'

        # ---- check dimensions of input data
        dims = A.dims
        shape = A.shape
        if(dims[0] != self.ncoldim or shape[0] != self.N):
            raise RuntimeError('(sph_zonal_mean_generic() Expected the first (leftmost) '\
                               'dimension of variable {} to be {} of length {}'.format(
                                                          A.name, self.ncoldim, self.N))

        # ---- get precision of input data
        prec_A = A.dtype

        # ---- extract and reshape data for averaging
        AA = A.values
        if(len(dims) == 1): DD = 1
        else:               DD = np.prod(shape[1:])  # outer dimension of data
        AA = AA.reshape((self.N, DD))
        self.logger.print('Reshaped variable {}: {} -> {}'.format(A.name, shape, AA.shape))
        
        # ---- do averaging, reshape
        self.logger.print('Taking zonal average of variable {}...'.format(A.name))
        Abar = np.matmul(np.matmul(Y, self.Y0inv), AA)
        
        oldshape = Abar.shape
        NN = Y.shape[0]         # outer dimension of Y (either N or M)
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
                              'ncol={} -> ncol={}'.format(A.name, self.N, self.M))
        A.values = Abar
        A.attrs['long_name'] = 'zonal mean of {}'.format(A.name)

        # ---- cast the data type of the zonal mean to that of the input data
        # since the matrices Y are stored as 64-bit floats, the matrix multiplication
        # step above will result in a 64-bit type for Abar. If this differs from the
        # type of the input data (e.g. float32), then cast the type of the result to 
        # match the input data
        A = A.astype(prec_A) 
        return A 
         
    def sph_zonal_mean_native(self, A):
        '''
        Calls sph_zonal_mean_generic with Y=Y0 (the native grid). See _sph_zonal_mean_generic
        docstrings for argument descriptions
        '''
        return self._sph_zonal_mean_generic(A, self.Y0)
    def sph_zonal_mean(self, A):
        '''
        Calls sph_zonal_mean_generic with Y=Y0p (the output grid). See _sph_zonal_mean_generic
        docstrings for argument descriptions
        '''
        return self._sph_zonal_mean_generic(A, self.Y0p)
    

    # --------------------------------------------------


    def sph_compute_matrices(self, overwrite=False, read_only=False, no_write=False):
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
        no_write : bool, optional
            If True, do not write remap matrices to file; compute_sph_matrices 
            will instead only return the matrices as 2D arrays to caller. 
            Defaults to False.
        '''
        
        self.logger.print('called sph_compute_matrices() for (M x N) '\
                          '= ({} x {}), L = {}'.format(self.M, self.N, self.L))
 
        # ---- read the averaging and remap matrices from file, if exists
        read_Y0 = False
        try:
            if(overwrite):
                if os.path.isfile(self.Y0_file_out): os.remove(self.Y0_file_out)
                if os.path.isfile(self.Y0p_file_out): os.remove(self.Y0p_file_out)
            Y0_ds  = xr.open_dataset(self.Y0_file_out)
            Y0p_ds = xr.open_dataset(self.Y0p_file_out)
            Y0    = Y0_ds['Y0'].values
            Y0inv = Y0_ds['Y0inv'].values
            Y0p   = Y0p_ds['Y0p'].values
            self.logger.print('Y0,Y0inv read from file {}'.format(self.Y0_file_out))
            self.logger.print('Y0\' read from file {}'.format(self.Y0p_file_out))
            read_Y0 = True
        except FileNotFoundError:
            if(read_only):
                return 

        # ---- if the matrices Y0, Y0inv, Y0p were not read from file, compute and 
        #      write them out now
        if(not read_Y0):
           
            # ---- place the grid area weights on the diagonal NxN matrix
            if(self.weights is not None):
                if(len(self.weights) != len(self.lat)):
                    raise RuntimeError('number of weights must equal number of native grid latitudes!')
                self.logger.print('building diag(w) from supplied weights...')
                self.diagw = np.diag(self.weights)

            # ---- compute zeroth-order spherical harmonics Y[l,m=0] at the input lats
            self.logger.print('building Y0...')
            Y0 = np.zeros((self.N, self.L+1)) # matrix to store spherical harmonics on input lats
            coalt = np.deg2rad(90 - self.lat) # the coaltitude (0 at NP, 180 at SP) in radians
            for ll in self.l:
                Y0[:,ll] = sph_harm(0, ll, 0, coalt).real

            # ---- compute zeroth-order spherical harmonics Y[l,m=0] at the output lats
            self.logger.print('building Y0\'...')
            Y0p = np.zeros((self.M, self.L+1))    # matrix to store spherical harmonics on input lats
            coalt = np.deg2rad(90 - self.lat_out) # the coaltitude (0 at NP, 180 at SP) in radians
            for ll in self.l:
                Y0p[:,ll] = sph_harm(0, ll, 0, coalt).real

            # ---- construct the remap matrices
            #
            # Y0*Y0inv recovers the zonal mean on a reduced grid with M data points
            # Y0p*Y0inv recovers the zonal mean on the native grid with N data points
            #
            # These lines are the bulk of the computation of this function; the matrix inversions and 
            # subsequent multplications are timed, and the results printed, if debug=True.
            # After solving, the matrices Y0 and Y0inv are written out to file. The format is NetCDF; 
            # for completeness, we name the variable, dimensions, and provide a long name.
           
            # ---- invert Y0
            if(self.weights is not None):
                self.logger.print('building inv(Y0) = Y^T diag(w)...', with_timer=True)
                Y0inv = np.matmul(Y0.T, self.diagw)
                self.logger.timer()
            else:
                self.logger.print('no grid weights provided; inverting Y0...', with_timer=True)
                Y0inv = lstsq(Y0, np.identity(self.N))[0]
                self.logger.timer()

            # ---- do quick sanity check; Y0inv*Y0 should give the identity matrix
            diagsum = np.sum(np.diagonal(np.matmul(Y0inv, Y0)))
            matsum  = np.sum(np.matmul(Y0inv, Y0)) - diagsum
            self.logger.print('Sanity check: sum(diag(Y0inv*Y0)) = {} '\
                              '(should be {})'.format(diagsum, self.L+1))
            self.logger.print('Sanity check: sum(offdiag(Y0inv*Y0)) = {} '\
                              '(should be zero)'.format(matsum))
        
            if(not no_write): 
                # -- write out Y
                Y0_da      = xr.DataArray(Y0, dims=('ncol','l'))
                Y0_da.name = 'Y0'
                Y0_da.attrs['long_name'] = 'Matrix Y0 for grid {}'.format(self.grid_name)
                Y0inv_da  = xr.DataArray(Y0inv, dims=('l','ncol'))
                Y0inv_da.name = 'Y0inv'
                Y0inv_da.attrs['long_name'] = 'Matrix Y0inv for grid {}'.format(self.grid_name)
                Y0_ds = xr.merge([Y0_da, Y0inv_da])
                Y0_ds.to_netcdf(self.Y0_file_out, encoding={'Y0':NCENC,'Y0inv':NCENC})
                self.logger.print('Y0,Y0inv wrote to file {}'.format(self.Y0_file_out))
            
                # -- write out Y'
                Y0p_da      = xr.DataArray(Y0p, dims=('ncol','l'))
                Y0p_da.name = 'Y0p'
                Y0p_da.attrs['long_name'] = 'Matrix Y0p for grid {}'.format(self.grid_out_name)
                Y0p_da.to_netcdf(self.Y0p_file_out, encoding={'Y0p':NCENC})       
                self.logger.print('Y0p wrote to file {}'.format(self.Y0p_file_out))
        
        # -- done; export to class namespace and return
        self.Y0    = Y0
        self.Y0inv = Y0inv
        self.Y0p   = Y0p


# -------------------------------------------------------------------------


