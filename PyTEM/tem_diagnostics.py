# Joe Hollowed
# University of Michigan 2023

# This module implements a class which provides an interface for computing TEM diagnostic 
# quantities as provided in Appendix A of DynVarMIP (Gerber+Manzini 2016). An object is 
# initialized with a dataset and several options for the calculations, which can then be
# used to recover various TEM diagnostics.


# =========================================================================


# ---- import dependencies
import warnings
import numpy as np
import xarray as xr

# ---- imports from this package
import util
from constants import *
from sph_zonal_mean import *

default_dimnames = {'horz':'ncol', 'vert':'lev', 'time':'time'}

# -------------------------------------------------------------------------


class TEMDiagnostics:

    def __init__(self, ua, va, ta, wap, p, zm_dlat=1, L=50, 
                 dim_names=default_dims, log_pressure=True, grid_name=None, 
                 zm_grid_name=None, map_save_dest=None, overwrite_map=False, 
                 debug=False):
        '''
        This class provides interfaces for computing TEM diagnostic quantities, assuming 
        either a log-pressure (the default) or pure-pressure vertical coordinate. Upon 
        initialization, the input data is checked and reshaped as necessary, zonal-averaging 
        matrices are built, and zonal mean and eddy components are computed. The available 
        data inputs and outputs (and their naming conventions) are derived from the DynVarMIP
        experimental protocol (Gerber+Manzini 2016).

        Parameters
        ----------
        ua : xarray DataArray
            Eastward wind, in m/s. This and all other data inputs are expected in at least 2D
            (unstructured spatially 3D), with optional third temporal dimension. The 
            horizontal, vertical, and optional temporal dimensions must be named, with names
            matching those given by the argument 'dim_names'. The expected units of the 
            coordinates are: lat in degrees, lev in hPa, time in hours.
        va : xarray DataArray
            Northward wind, in m/s.
        ta : xarray DataArray
            Air temperature, in K.
        wap : xarray DataArray
            Vertical pressure velocity, in Pa/s.
        p : xarray DataArray
            Air pressure, in Pa.
        zm_dlat : float, optional
            Spacing used to generate uniform 1D latitude grid for zonal means, in degrees. 
            Defaults to 1 degree.
        L : int, optional
            Maximum spherical harmonic degree to use in zonal averaging. Defaults to 50.
        dim_names : dict, optional
            Dimension names to expect from the input data, as a dictionary that contains
            the following key:value pairs:
            'horz':{name of horizontal dimension}
            'vert':{name of vertical dimension}
            'time':{name of temporal dimension}
            Defaults to: {'horz':'ncol', 'vert':'lev', 'time':'time'}. The order of the
            dimensions in the input data does not matter; data will be reshaped at 
            object initialization.
        log_pressure : bool, optional
            Whether or not to perform the TEM analysis in a log-pressure vertical 
            coordinate.If False, then a pure pressure vertical coordinate is used. 
            Defaults to True.
        grid_name : str, optional
            The 'grid_name' input argument to sph_zonal_mean(). See docstrings therein.
        zm_grid_name : str, optional
            The 'grid_out_name' input argument to sph_zonal_mean(). See docstrings therein.
        map_save_dest : str, optional
            The 'save_dest' input argument to sph_zonal_mean(). See docstrings therein.
        overwrite_map : bool, optional
            The 'overwrite' input argument to sph_zonal_mean(). See docstrings therein.
        debug : bool, optional
        
        Methods
        -------
        
        '''

        self.logger = util.logger(debug)
        
        # ---- get input args
        # variables
        self.p    = p    # pressure [Pa]
        self.ua   = ua   # eastward wind [m/s]
        self.va   = va   # northward wind [m/s]
        self.ta   = ta   # temperature [K]
        self.wap  = wap  # vertical pressure velocity [Pa/s]
        # options
        self.L         = L        # max. spherical harmonic degree 
        self.debug     = debug    # logging bool flag
        self.zm_dlat   = zm_dlat  # zonal mean discretization spacing [deg]
        self.log_pres  = log_pres # log pressure bool flag
        
        # ---- declare new fields
        self.zm_lat = None     # zonal mean latitude set [deg]
        self.theta = None      # potential temperature [K]
        self.up = None         # "u-prime", zonally asymmetric anomaly in ua
        self.vp = None         # "v-prime", zonally asymmetric anomaly in va
        self.wapp = None       # "wap-prime", zonally asymmetric anomaly in wap
        self.thetap = None     # "theta-prime", zonally asymmetric anomaly in theta
        self.ub = None         # "u-bar", zonal mean of ua
        self.vb = None         # "v-bar", zonal mean of va
        self.wapb = None       # "wap-bar", zonal mean of wap
        self.thetab = None     # "theta-bar", zonal mean of theta
        self.upvp = None       # product of up and vp, aka northward flux of eastward momentum
        self.upvpb = None      # zonal mean of upvp
        self.upwapp = None     # product of up and wp, aka upward flux of eastward momentum
        self.upwwapb = None    # zonal mean of upwpb
        self.vptp = None       # product of vp and thetap, aka northward flux of potential temperature
        self.vptpb = None      # zonal mean of vptpb
        self._epfy = None      # northward component of the Eliassen-Palm flux [m3/s2] 
        self._epfz = None      # upward component of the Eliassen-Palm flux [m3/s2]
        self._epdiv = None     # divergence of the Eliassen-Palm flux (this output not in DynVarMIP)
        self._vtem = None      # TEM (residual) northward wind [m/s]
        self._wtem = None      # TEM (residual) upward wind [m/s]
        self._psitem = None    # TEM mass stream function [kg/s]
        self._utendepfd = None # tendency of eastward wind due to EP flux divergence [m/s]
        self._utendvtem = None # tendency of eastward wind due to TEM northward wind advection + coriolis [m/s]
        self._utendwtem = None # tendency of eastward wind due to TEM upward wind advection [m/s]
         
        # ---- handle dimensions of input data, generate coordinate arrays
        self.handle_dims()
        
        # ---- construct zonal averaging obeject
        self.logger.print('Getting zonal averaging matrices...')
        zm = sph_zonal_averager(self.lat, self.zm_lat, self.L, grid_name, zm_grid_name, map_save_dest)
        zm.sph_zm_matrices(overwrite_map, self.debug)
        
        # ---- get potential temperature
        self.compute_potential_temperature()

        # ---- get zonal means, eddy components
        self.decompose_zm_eddy()
        
        # ---- get fluxes
        self.compute_fluxes()


    # --------------------------------------------------


    def handle_dims(self):
        '''
        Checks that input data dimensions are as expected. Adds temporal dimensions if needed.
        Transposes input data such that data shapes are consistent with (ncol, lev, time). Builds
        latitude discretization for zonal means.
        '''
        
        # ---- get dim names        
        self.dim_names = dim_names
        self.ncoldim   = self.dim_names['horz']
        self.levdim    = self.dim_names['vert']
        self.timedim   = self.dim_names['time']
        self.data_dims = (self.ncoldim, self.levdim, self.timedim)
        
        allvars     = {'ua':self.ua, 'va':self.va, 'ta':self.ta, 'wap':self.wap, 'p':self.p}
        allcoords   = {'lat':self.lat, 'lev':self.lev, 'time':self.time}
        alldat      = dict(allvars, **allcoords)
        
        # ---- verify dat input format, transform as needed
        for var,dat in alldat.items():
            if(not isinstance(dat, xarray.core.dataarray.Dataset)):
                raise RuntimeError('Input data for arg \'{}\' must be an xarray DataArray'.format(var)
            # check data dimensions
            if(self.ncoldim not in dat.dims):
                raise RuntimeError('Input data does not contain dim {}'.format(self.ncoldim))
            if(len(dat.dims) < 2 or len(dat.dims)):
                raise RuntimeError('Input data has {0} dims, expected either 2 ({1}, {2}) '\
                                   'or 3 ({1}, {2}, {3})'.format(len(dat.dims), self.ncoldim, 
                                                                 self.levdim, self.timedim)
            # if time doesn't exist, expand the data arrays to include a new temporal
            # dimension of length 1, value 0.
            if(self.tiemdim not in dat.dims):
                dat = dat.expand_dims(self.tiemdim, axis=len(dat.dims))
                self.logger.print('Expanded variable {} to contain new dimension \'{}\''.format(
                                                                              var, self.timedim))
            # reshape data as (ncol, lev, time)
            old_dims = dat.dims
            dat = dat.transpose(*self.data_dims)
            self.logger.print('Variable {} transposed: {} -> {}'.format(var, self.ncoldim, 
                                                                        old_dims, dat.dims))
        # ---- get coordinates, data lengths
        self.lat  = ua[self.ncoldim]  # latitude [deg]
        self.lev  = ua[self.levdim]   # model levels [hPa]
        self.time = ua[self.timedim]  # time positions [hours]
        self.NCOL = self.ua.shape[0]
        self.NLEV = self.ua.shape[1]  
        self.NT   = self.ua.shape[2]  
        
        # ---- build uniform latitude discretization zm_lat for zonal mean remap
        tol = 1e-6                  # tolerance for float coparisons
        assert (180/self.zm_dlat).is_integer(), '180 must be divisible by dlat_out'
        self.zm_lat = np.arange(-90, 90+self.zm_dlat, self.zm_dlat)
        if(self.zm_lat[-1] > 90+tol): self.zm_lat = self.zm_lat[:-1]
        self.ZM_N = len(self.zm_lat)   # length of zonal mean latitude set
        self.logger.print('Built zonal-mean latitude grid with {} deg spacing: {}'.format(
                                                                self.zm_dlat, self.zm_lat))
    

    # --------------------------------------------------
    

    def compute_potential_temperature():
        '''
        Computes the potential temperature from temperature and pressure
        '''
        self.theta = self.ta * (p0/self.p)**k
        self.theta.name = 'THETA'


    # --------------------------------------------------
    

    def decompose_zm_eddy():
        '''
        Decomposes atmospheric variables into zonal means and eddy components.
        ''' 
        self.ub =  
    

    # --------------------------------------------------
    

    def compute_fluxes():
        '''
        Computes momentum and potential temperature fluxes, and their zonal averages
        '''
    

    # --------------------------------------------------


    def momentum_budget():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''

        self.epfy = 
    
    # --------------------------------------------------

    def epfy():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
    
    # --------------------------------------------------

    def epfz():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
         pass
    
    # --------------------------------------------------

    def epfz():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
         pass
    
    # --------------------------------------------------

    def vtem():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
         pass
    
    # --------------------------------------------------

    def wtem():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
         pass
    
    # --------------------------------------------------

    def psitem():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
        pass
    
    # --------------------------------------------------
 
    def utendepfd():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
        pass
    
    # --------------------------------------------------
 
    def utendvtem():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
         pass
    
    # --------------------------------------------------

    def utendwtem():
        '''
        Desc

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    
