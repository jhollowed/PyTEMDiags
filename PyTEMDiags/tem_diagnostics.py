# Joe Hollowed
# University of Michigan 2023

# This module implements a class which provides an interface for computing TEM diagnostic 
# quantities as provided in Appendix A of DynVarMIP (Gerber+Manzini 2016). An object is 
# initialized with a dataset and several options for the calculations, which can then be
# used to recover various TEM diagnostics.


# =========================================================================


# ---- import dependencies
import pdb
import warnings
import numpy as np
import xarray as xr

# ---- imports from this package
from . import tem_util as util
from .constants import *
from .sph_zonal_mean import *

# ---- global constants
DEFAULT_DIMS = {'horz':'ncol', 'vert':'lev', 'time':'time'}


# -------------------------------------------------------------------------


class TEMDiagnostics:
    def __init__(self, ua, va, ta, wap, p, q, lat_native, p0=P0, zm_dlat=1, L=150, 
                 lat_weights=None, dim_names=DEFAULT_DIMS, log_pressure=True, 
                 grid_name=None, zm_grid_name=None, map_save_dest=None, 
                 overwrite_map=False, zm_pole_points=False, debug=False):
        '''
        This class provides interfaces for computing TEM diagnostic quantities, assuming 
        either a log-pressure (the default) or pure-pressure vertical coordinate. Upon 
        initialization, the input data is checked and reshaped as necessary, zonal-averaging 
        matrices are built, and zonal mean and eddy components are computed. The available 
        data inputs and outputs (and their naming conventions) are derived from the DynVarMIP
        experimental protocol (Gerber+Manzini 2016). Funcitonality consistent with these 
        conventions is also included for computin the tracer TEM terms as described in 
        Abalos+ 2017.

        Parameters
        ----------
        ua : xarray DataArray
            Eastward wind, in m/s. This and all other data inputs are expected in at least 2D
            (unstructured spatially 3D), with optional third temporal dimension. The 
            horizontal, vertical, and optional temporal dimensions must be named, with names
            matching those given by the argument 'dim_names'. The expected units of the 
            coordinates are: ncol dimensionless, lev in hPa, time in hours.
        va : xarray DataArray
            Northward wind, in m/s.
        ta : xarray DataArray
            Air temperature, in K.
        wap : xarray DataArray
            Vertical pressure velocity, in Pa/s.
        p : xarray DataArray
            Air pressure, in Pa. This variable can either be a coordinate or a variable.
            If a coordinate, requires:
                1D array, matching the length of the data variables in the vertical dimension
            If a variable, requires:
                ND array, matching the dimensionality of the data.
            If p is passed as a variable, then all derivatives are computed individually 
            per-column, and an extra remapping step is performed to return zonally avergaed 
            quantities, thus the computations are slower.
        q : xarray DataArray, or list of xarray DataArray
            dimensionless tracer mass mixing ratios (e.g. kg/kg). This arugment can support any
            arbitrary number of tracer species. If passed as a list, then the list elements 
            should each be xarray DataArrays, each corresponding to a unique tracer. Dimenionality
            and units of the arrays should match the description of argument ua.
        lat_native : xarray DataArray
            Latitudes in degrees.
        lat_weights : xarray DataArray
            Grid cell areas corresponding to grid cells at latitudes lat_native, in any units.
        p0 : float
            Reference pressure in Pa. Defaults to the DynVarMIP value in constants.py
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
            The 'grid_name' input argument to sph_zonal_averager(). See docstrings therein.
        zm_grid_name : str, optional
            The 'grid_out_name' input argument to sph_zonal_averager(). See docstrings therein.
        map_save_dest : str, optional
            The 'save_dest' input argument to sph_zonal_averager(). See docstrings therein.
        overwrite_map : bool, optional
            The 'overwrite' input argument to sph_compute_matrices(). See docstrings therein.
        zm_pole_points : bool, optional
            Whether or not to include the pole points in the zonal mean latitude set 
            specified by zm_dlat. If False, then after the uniform lattiude set with 
            spacing zm_dlat is built, it will then be shifted such that the data points
            lie at the midpoint of each of these cells. Defaults to False.
        debug : bool, optional
            Whether or not to print verbose progress statements to stdout.
        
        Public Methods
        --------------
        vtem()
            Returns the TEM northward wind in m/s.
        wtem()
            Returns the TEM upward wind in m/s.
        psitem()
            Returns the TEM mass stream function in kg/s.
        epfy()
            Returns the northward component of the EP flux in m3/s2.
        epfz()
            Returns the upward component of the EP flux in m3/s2.
        epdiv()
            Returns the EP flux divergence.
        utendepfd()
            Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        utendvtem()
            Returns the tendency of eastward wind due to TEM northward wind advection 
            and coriolis in m/s2.
        utendwtem()
            Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        etfy()
            Returns the northward component of the eddy tracer flux in m2/s.
        etfz()
            Returns the upward component of the eddy tracer flux in m2/s.
        etdiv()
            Returns the eddy tracer flux divergence.
        qtendetfd()
            Returns the tendency of tracer mixing ratios due to eddy tracer flux divergence in 1/s.
        qtendvtem()
            Returns the tendency of tracer mixing ratios due to TEM northward wind advection 
            and coriolis in 1/s.
        qtendwtem()
            Returns the tendency of tracer mixing ratios due to TEM upward wind advection in 1/s.
        
        Public Attributes
        -----------------
        All input arguments are attributes of the same name, as well as:
        ZM : sph_zonal_averager object
            Zonal averager object associated with current configuration.
        lat_zm : 1D array
            Zonal mean latitude set in degrees.
        theta : N-D array
            Potential temperature in K, matching dims of input ta.
        up : N-D array
            "u-prime", zonally asymmetric anomaly in ua in m/s, matching dims of input ua.
        vp : N-D array
            "v-prime", zonally asymmetric anomaly in va in m/s, matching dims of input va.
        wapp : N-D array
            "wap-prime", zonally asymmetric anomaly in wap in Pa/s, matching dims of 
            input wap.
        thetap : N-D array
            "theta-prime", zonally asymmetric anomaly in K, matching dims of input ta.
        qp : N-D array
            "q-prime", zonally asymmetric anomaly in kg/kg, matching dims of input q.
        ub : N-D array
            "u-bar", zonal mean of ua in m/s, matching length of lat_zm in the horizontral, 
            otherwise matching dims of input ua.
        vb : N-D array
            "v-bar", zonal mean of va in m/s, matching length of lat_zm in the horizontral, 
            otherwise matching dims of input va.
        wapb : N-D array
            "omega-bar", zonal mean of wap in Pa/a, matching length of lat_zm in the 
            horizontral, otherwise matching dims of input wap.
        thetab : N-D array
            "theta-bar", zonal mean of theta in K, matching length of lat_zm in the 
            horizontral, otherwise matching dims of input ta.
        qb : N-D array
            "q-bar", zonal mean of q in kg/kg, matching length of lat_zm in the 
            horizontral, otherwise matching dims of input q.
        upvp : N-D array
            Product of up and vp in m2/s2, aka northward flux of eastward momentum, matching 
            dims of input ua.
        upvpb : N-D array
            Zonal mean of upvp in m2/s2, matching length of lat_zm in the horizontal, 
            otherwise matching dims of input ua.
        upwapp : N-D array
            Product of up and wapp in (m Pa)/s2, aka upward flux of eastward momentum, 
            matching dims of input ua.
        upwwappb : N-D array
            Zonal mean of upwapp in (m Pa)/s2, matching length of lat_zm in the horizontal, 
            otherwise matching dims of input ua.
        vptp : N-D array       
            Product of vp and thetap in (m K)/s, aka northward flux of potential temperature,
            matching dims of input va.
        vptpb : N-D array
            Zonal mean of vptp in (m K)/s, matching length of lat_zm in the horizontal, 
            otherwise matching dims of input va.
        qpvp : N-D array
            Product of qp and vp in m/s, aka northward eddy tracer flux, matching 
            dims of input q.
        qpwapp : N-D array
            Product of qp and wapp in Pa/s, aka upward eddy tracer flux, 
            matching dims of input q.
        qpvpb : N-D array
            Zonal mean of qpvp in m/s, matching length of lat_zm in the horizontal, 
            otherwise matching dims of input q.
        qpwappb : N-D array
            Zonal mean of qpwapp in Pa/s, matching length of lat_zm in the horizontal, 
            otherwise matching dims of input q.
        out_file : str
            Output filename, used for naming data files written by the to_netcdf() method. 
            Simple string that contains metadata about the current TEM option configuration.
        '''

        self._logger = util.logger(debug, header=True)
        
        # ---- get input args
        # variables
        self.p     = p    # pressure [Pa]
        self.ua    = ua   # eastward wind [m/s]
        self.va    = va   # northward wind [m/s]
        self.ta    = ta   # temperature [K]
        self.wap   = wap  # vertical pressure velocity [Pa/s]
        self.p0    = p0   # reference pressure [Pa]
        self.q     = q    # tracer mixing ratios [kg/kg]
        self.ntrac = None # number of input tracers
        self.lat_native  = lat_native  # latitudes [deg]
        self.lat_weights = lat_weights  # latitude weights (grid cell areas)
        # options
        self.L              = L
        self.debug          = debug
        self.zm_dlat        = zm_dlat
        self.log_pres       = log_pressure
        self.dim_names      = dim_names
        self.zm_pole_points = zm_pole_points
        self.grid_name      = grid_name
        self.zm_grid_name   = zm_grid_name
        self.map_save_dest  = map_save_dest
        self.overwrite_map  = overwrite_map
        self._ptype         = None  # pressure type, either 'coord' or 'var'; will be set 
                                    # in config_dims()
          
        # ---- veryify input data dimensions, configure data and settings
        self._config_dims()
        
        # ---- construct zonal averaging obeject
        self._logger.print('Getting zonal averaging matrices...')
        self.ZM = sph_zonal_averager(self.lat_native, self.lat_zm, self.L, 
                                     weights=self.lat_weights,  
                                     grid_name=grid_name, grid_out_name=zm_grid_name, 
                                     save_dest=map_save_dest, debug=debug, 
                                     overwrite=overwrite_map)
        if(self.ZM.Y0 is None or self.ZM.Y0p is None):
            self.ZM.sph_compute_matrices(overwrite=overwrite_map)

        # ---- configure computational fucntions for various operations based on inputs
        self._config_functions()
        
        # ---- get potential temperature
        self._compute_potential_temperature()

        # ---- get zonal means, eddy components
        self._decompose_zm_eddy()
        
        # ---- compute fluxes, vertical and meridional derivaties, streamfunction terms
        self._compute_fluxes()
        self._compute_derivatives()
        
        # ---- output filename used by self.to_netcdf()
        self._out_file = None
     
    # --------------------------------------------------

    def _config_dims(self):
        '''
        Checks that input data dimensions are as expected. Adds temporal dimensions 
        if needed. Transposes input data such that data shapes are consistent with 
        (ncol, lev, time). Builds latitude discretization for zonal means.
        '''

        # ---- get dim names
        self.ncolname   = self.dim_names['horz']
        self.levname    = self.dim_names['vert']
        # use default time dim name if not present in data
        try: self.timename   = self.dim_names['time']
        except KeyError: self.timename = DEFAULT_DIMS['time']
        self.data_dims = (self.ncolname, self.levname, self.timename)

        # ---- ensure tracers are a list of DataArrays
        bad_q = False
        if(type(self.q) is not type(list())):
            # if q isn't a list, then only a single tracer was passed, and thus
            # it should be a DataArray. Place this variable into a list of length-1
            if(type(self.q) is not xr.core.dataarray.DataArray): bad_q = True
            else: self.q = [self.q]
        else:
            # if q is a list, multiple tracers were passed. All should be DataArrays
            for qi in self.q:
                if(type(qi) is not xr.core.dataarray.DataArray): bad_q = True
        if(bad_q): 
            raise RuntimeError('tracers q must be passed as an xarray DataArray, or'\
                               'a list of xarray DataArrays')
        self.ntac = len(self.q) # record number of input tracers
        self._logger.print('Number of input tracers: {}'.format(self.ntrac))

        # ---- gather variables for dimensions config 
        allvars    = {'ua':self.ua, 'va':self.va, 'ta':self.ta, 
                      'wap':self.wap, 'lat':self.lat_native}
        alltracers = dict(('q{}'.format(i), self.q[i]), for i in range(len(self.q)))
        allvars    = {**allvars, **alltracers}

        # ---- check if pressure was input as a coordinate (1D) or a variable (>1D)
        if len(self.p.dims) == 1 and self.p.dims[0] == self.levname:
            self._ptype = 'coord'
        elif self.p.dims == self.ua.dims and self.levname in self.p.dims:
            self._ptype = 'var'
            allvars['p'] = self.p
        else:
            raise RuntimeError('pressure p must be input as either a 1D coordinate matching '\
                               'the vertical dimension of the data in length (currently {}), '\
                               'or a variable with dimensionality matching the data '\
                               '(currently {})'.format(
                                self.ua.shape[self.ua.dims.index(self.levname)], self.ua.dims))
            
        # ---- verify dat input format, transform as needed
        for var,dat in allvars.items():
            if(not isinstance(dat, xr.core.dataarray.DataArray)):
                raise RuntimeError('Input data for arg \'{}\' must be an '\
                                   'xarray DataArray'.format(var))
            # check data dimensions
            if(self.ncolname not in dat.dims):
                raise RuntimeError('Input data {} does not contain dim {}'.format(
                                                                var, self.ncolname)) 
            # ensure passed lat and data ncol are consistent
            if(dat.shape[dat.dims.index(self.ncolname)] != len(self.lat_native)):
                raise RuntimeError('Dimension {} in variable {} is length {}, but input '\
                                   'parameter lat is length {}; these must match!'.format(
                                           self.ncolname, var, var.shape[0], len(self.lat)))
            if(var == 'lat'): continue
            # input must have 2 or 3 dimensions 
            if(len(dat.dims) < 2 or len(dat.dims) > 3):
                raise RuntimeError('Input data has {0} dims, expected either 2 ({1}, {2}) '\
                                   'or 3 ({1}, {2}, {3})'.format(len(dat.dims), self.ncolname, 
                                                                 self.levname, self.timename))
            # if time doesn't exist, expand the data arrays to include a new temporal
            # dimension of length 1, value 0.
            if(self.timename not in dat.dims):
                dat = dat.expand_dims(self.timename, axis=len(dat.dims))
                self._logger.print('Expanded variable {} to contain new '\
                                  'dimension \'{}\''.format(var, self.timename))

        # ---- reshape data as (ncol, lev, time)
        # data is now verified to meet the expected input criteria; 
        # next transpose data to standard format with ncol as the first dimension
        # (this must be done outside of the loop above to avoid allocating memory 
        #  for copies of the underlying data objects in the dict allvars, I think)
        old_dims = self.ua.dims
        self.ua = self.ua.transpose(*self.data_dims)
        self._logger.print('Variable ua transposed: {} -> {}'.format(old_dims, self.ua.dims))
        old_dims = self.va.dims
        self.va = self.va.transpose(*self.data_dims)
        self._logger.print('Variable va transposed: {} -> {}'.format(old_dims, self.va.dims))
        old_dims = self.ta.dims
        self.ta = self.ta.transpose(*self.data_dims)
        self._logger.print('Variable ta transposed: {} -> {}'.format(old_dims, self.ta.dims))
        old_dims = self.wap.dims
        self.wap = self.wap.transpose(*self.data_dims)
        self._logger.print('Variable wap transposed: {} -> {}'.format(old_dims, self.wap.dims))
        if(self._ptype == 'var'):
            old_dims = self.p.dims
            self.p = self.p.transpose(*self.data_dims)
            self._logger.print('Variable p transposed: {} -> {}'.format(old_dims, self.p.dims))
        for(i in range(len(self.q))):
            old_dims = self.q[i].dims
            self.q[i] = self.q[i].transpose(*self.data_dims)
            self._logger.print('Variable q[{}] transposed: {} -> {}'.format(
                                                i, old_dims, self.q[i].dims))

        
        # ---- get coordinates, data lengths
        self.lev  = self.ua[self.levname]   # model levels [hPa]
        self.time = self.ua[self.timename]  # time positions [hours]
        self.NCOL = self.ua.shape[0]
        self.NLEV = self.ua.shape[1]  
        self.NT   = self.ua.shape[2] 
        self._logger.print('DATA DIMENSIONS: {} x {} x {} = {} x {} x {}'.format(
                                                   self.ncolname, self.levname, self.timename, 
                                                   self.NCOL, self.NLEV, self.NT))
        
        # ---- build uniform latitude discretization lat_zm for zonal mean remap
        tol = 1e-6                  # tolerance for float coparisons
        assert (180/self.zm_dlat).is_integer(), '180 must be divisible by dlat_out'
        
        self.lat_zm = np.arange(-90, 90+self.zm_dlat, self.zm_dlat)
        if(self.lat_zm[-1] > 90+tol): self.lat_zm = self.lat_zm[:-1]
        if(not self.zm_pole_points):
            self.lat_zm = (self.lat_zm[1:] + self.lat_zm[:-1]) / 2
        
        self.ZM_N = len(self.lat_zm)   # length of zonal mean latitude set
        self._logger.print('Built zonal-mean latitude grid with {} deg spacing: {}'.format(
                                                                self.zm_dlat, self.lat_zm))
        
        # --- get latitude-based quantities
        self._f_native = 2*Om*np.sin(self.lat_native * np.pi/180) # coriolis parameter on native grid
        self._f_zm     = 2*Om*np.sin(self.lat_zm * np.pi/180)     # coriolis parameter on zonal mean grid
        self._coslat_native = np.cos(self.lat_native * np.pi/180) # cosine of latitude on native grid
        self._coslat_zm     = np.cos(self.lat_zm * np.pi/180)     # cosine of latitude on zonal mean grid

        # ---- transform pressure p to a coordinate if needed
        # If the pressure was input as a 3D variable, but the pressure is unfiorm on each
        # vertical level for all horizontal columns and all time, then convert to a 1D coordinate 
        # (see more in the code section below)
        make_p_coord = True
        for k in range(self.NLEV):
            p = self.p[0,k,0]
            if(np.sum(self.p[:,k,:] == p) != self.NCOL*self.NT):
                self._logger.print('pressure is not uniform at level {} (reference pressure '\
                                   '{:.2f}); retaining as 3D variable'.format(
                                                     k, self.ua.lev.values[k]))
                make_p_coord = False
                break
        if(make_p_coord):
            self.p = self.p[0,:,0]
            self._ptype = 'coord'
            self._logger.print('reduced pressure p from a 3D {} variable to a 1D (length {}) '\
                              'coordinate'.format(self.ua.shape, len(self.p)))

        # ---- check pressure direction convention
        # ensure that pressure increases toward right-end of arrays. If not, flip
        # this axis for all data
        if(self.lev[0] > self.lev[-1]):
            self.ua  = self.ua.reindex({self.levname, self.lev[::-1]})
            self.vai = self.va.reindex({self.levname, self.lev[::-1]})
            self.ta  = self.ta.reindex({self.levname, self.lev[::-1]})
            self.wap = self.wap.reindex({self.levname, self.lev[::-1]})
            self.p   = self.p.reindex({self.levname, self.lev[::-1]})
            for i in range(len(slf.q)):
                self.q[i] = self.q[i].reindex({self.levname, self.lev[::-1]}) 
            self.lev = self.ua[self.levname]
            self._logger.print('Reversed direction of vertical dimension for all data '\
                               '(such that the model top is the leftmost entry in the '\
                               'pressure data array)')
            
    # --------------------------------------------------
    
    def _config_functions(self):
        '''
        Depending on ptype (whether pressure p was input as a coordinate or a variable), 
        set different functions for various operations, including taking zonal means, 
        returning zonal mean attributes, multiplying data matrices by coordinates. Specifically:

         If pressure p was input as a 1D vertical coordinate, then:
          - (a) vertical derivatives and integrals are done with structured coordinates, 
                and thus
          - (b) zonally-averaged quantities need not be computed on the native grid, and 
                thus can be stored on the zonal-mean grid with latitudes self._lat_zm
          - (c) because of this, mutltiplications and deriviatives of latitude-based 
                quantities will use the latitude set self._lat_zm
          - (d) when zonal-mean quantities are externally accessed, they can be returned 
                without modification
          - (e) multiplying 3D variables (meaning either native or zonally-averaged, since 
                horizontal ncol is a single dimension) by 1D pressure requires proper 
                broadcasting
         If pressure p was input as a 3D variable, then:
          - (f) vertical derivatives must be computed per-column, thus
          - (g) zonally-averaged quantities need to be computed and stored on the native grid 
                with latitudes self._lat_native
          - (h) because of this, mutltiplications and deriviatives of latitude-based 
                quantities will use the latitude set self._lat_native
          - (i) when zonal-mean quantities are externally accessed, they must be remapped
            to the specified uniform latitude grid before return
          - (j) multiplying 3D variables (meaning either native or zonally-averaged, since 
                horizontal ncol is a single dimension) by pressure is done element-wise
         If pressure is either a coordinate or variable, then:
          - (k) multiplying 3D variables (meaning either native or zonally-averaged, since
                horizontal ncol is a single dimension) by 1D latitude requires proper 
                broadcasting
        '''

        if(self._ptype == 'coord'):
            self._p_gradient       = util.p_gradient_1d            # (a)
            self._p_integral       = util.p_integral_1d            # (a)
            self._zonal_mean       = self.ZM.sph_zonal_mean        # (b)
            self.lat, self.coslat  = self.lat_zm, self._coslat_zm  # (c)
            self.f                 = self._f_zm                    # (c)
            self._zm_return_func   = lambda x: x                   # (d)
            self._multiply_pres    = util.multiply_p_1d            # (e)
            self._logger.print('configuration set for pressure p as a coordinate')
        elif(self._ptype == 'var'):
            self._p_gradient       = util.p_gradient_3d                   # (f)
            self._p_integral       = util.p_integral_3d                   # (f)
            self._zonal_mean       = self.ZM.sph_zonal_mean_native        # (g)
            self.lat, self.coslat  = self.lat_native, self._coslat_native # (h)
            self.f                 = self._f_native                       # (h)
            self._zm_return_func   = self.ZM.sph_zonal_mean               # (i)
            self._multiply_pres    = util.multiply_p_3d                   # (j)
            self._logger.print('configuration set for pressure p as a variable')
        self._multiply_lat = util.multiply_lat #(k)
    
    # --------------------------------------------------

    '''
    Below are a set of getter functions for all zonally-averaged quantities.
    If ptype == cord, then these do nothing other than return the variables.
    If ptype == var, then:
    Internal to the class, these quantities are defined on the native grid, 
    where all computations occur (this is done since the vertical coordinate 
    is unstructured in pressure; computing everything on the native grid avoids 
    needing to assume model level pressure, or interpolate to pressure levels).
    It would be obnoxious, however, to return to the user zonal averages with 
    NCOL data points. Instead, these getter functions remap the quantities to
    lat_zm.
    '''
    @property
    def ub(self): return self._zm_return_func(self._ub)
    @property
    def vb(self): return self._zm_return_func(self._vb)
    @property
    def thetab(self): return self._zm_return_func(self._thetab)
    @property
    def wapb(self): return self._zm_return_func(self._wapb)
    @property
    def qb(self): return self._zm_return_func(self._qb)
    @property
    def up(self): return self._up
    @property
    def vp(self): return self._vp
    @property
    def thetap(self): return self._thetap
    @property
    def wapp(self): return self._wapp
    @property
    def qp(self): return self._qp
    @property
    def upvp(self): return self._upvp
    @property
    def upwapp(self): return self._upwapp
    @property
    def vptp(self): return self._vptp
    @property
    def qpvp(self): return self._qpvp
    @property
    def qpwapp(self): return self._qpwapp    
    @property
    def upvpb(self): return self._zm_return_func(self._upvpb)
    @property
    def upwappb(self): return self._zm_return_func(self._upwappb)
    @property
    def vptpb(self): return self._zm_return_func(self._vptpb)
    @property
    def qpvpb(self): return self._zm_return_func(self._qpvpb)
    @property
    def qpwappb(self): return self._zm_return_func(self._qpwappb)
    @property
    def dub_dp(self): return self._zm_return_func(self._dub_dp)
    @property
    def dthetab_dp(self): return self._zm_return_func(self._dthetab_dp)
    @property
    def ubcoslat(self): return self._zm_return_func(self._ubcoslat)
    @property
    def dubcoslat_dlat(self): return self._zm_return_func(self._dubcoslat_dlat)
    @property
    def psicoslat(self): return self._zm_return_func(self._psicoslat)
    @property
    def dpsicoslat_dlat(self): return self._zm_return_func(self._dpsicoslat_dlat)
    @property
    def dqb_dp(self): return self._zm_return_func(self.dqb_dp)
    @property
    def qbcoslat(self): return self._zm_return_func(self._qbcoslat)
    @property
    def dqbcoslat_dlat(self): return self._zm_return_func(self._dqbcoslat_dlat)
    @property
    def int_vbdp(self): return self._zm_return_func(self._int_vbdp) 
    @property
    def psi(self): return self._zm_return_func(self._psi)
    @property
    def dpsi_dp(self): return self._zm_return_func(self._dpsi_dp)
    @property
    def out_file(self): 
        if(self._out_file is None):
            warnings.warn('\'out_file\' is not set until to_netcdf() is called')
        return self._out_file
    
    def vtem(self): return self._zm_return_func(self._vtem())
    def omegatem(self): return self._zm_return_func(self._omegatem())
    def wtem(self): return self._zm_return_func(self._wtem())
    def omegatem(self): return self._zm_return_func(self._omegatem())
    def psitem(self): return self._zm_return_func(self._psitem())
    def epfy(self): return self._zm_return_func(self._epfy())
    def epfz(self): return self._zm_return_func(self._epfz())
    def epdiv(self): return self._zm_return_func(self._epdiv())
    def utendepfd(self): return self._zm_return_func(self._utendepfd())
    def utendvtem(self): return self._zm_return_func(self._utendvtem())
    def utendwtem(self): return self._zm_return_func(self._utendwtem())
    def etfy(self): return self._zm_return_func(self._epfy())
    def etfz(self): return self._zm_return_func(self._epfz())
    def etdiv(self): return self._zm_return_func(self._epdiv())
    def qtendetfd(self): return self._zm_return_func(self._utendepfd())
    def qtendvtem(self): return self._zm_return_func(self._utendvtem())
    def qtendwtem(self): return self._zm_return_func(self._utendwtem())

    # --------------------------------------------------
    
    def _compute_potential_temperature(self):
        '''
        Computes the potential temperature from temperature and pressure.
        '''
        self._logger.print('computing potential temperature...')

        # θ = T * (p0/p)**k
        theta = self._multiply_pres(self.ta, (self.p0/self.p)**k)
       
        # inherit coords, attributes of temperature
        self.theta = self.ta.copy(deep=True)
        self.theta.values = theta
        self.theta.name = 'THETA'
        self.theta.attrs['long_name'] = 'potential temperature'
        try: del self.theta.attrs['standard_name']
        except KeyError: pass

    # --------------------------------------------------

    def _decompose_zm_eddy(self):
        '''
        Decomposes atmospheric variables into zonal means and eddy components.
        ''' 
        self._logger.print('computing zonal means for input variables...')
        self._ub          = self._zonal_mean(self.ua)
        self._ub.name     = 'ub'
        self._up          = self.ua - self.ZM.sph_zonal_mean_native(self.ua)
        self._up.name     = 'up'
        self._vb          = self._zonal_mean(self.va) 
        self._vb.name     = 'vb'
        self._vp          = self.va - self.ZM.sph_zonal_mean_native(self.va)
        self._vp.name     = 'vp'
        self._thetab      = self._zonal_mean(self.theta) 
        self._thetab.name = 'thetab'
        self._thetap      = self.theta - self.ZM.sph_zonal_mean_native(self.theta)
        self._thetap.name = 'thetap'
        self._wapb        = self._zonal_mean(self.wap) 
        self._wapb.name   = 'wapb'
        self._wapp        = self.wap - self.ZM.sph_zonal_mean_native(self.wap)
        self._wapp.name   = 'wapp'
        self._logger.print('computing zonal means for tracers...')
        self._qb, self.qp = [None]*self.ntrac, [None]*welf.ntrac
        for i in range(len(self.ntrac)): 
            self._qb[i]       = self._zonal_mean(self.q[i])
            self._qb[i].name  = 'qb'
            self._qp[i]       = self.q[i] - self.ZM.sph_zonal_mean_native(self.q[i])
            self._qp[i].name  = 'qp'
    
    # --------------------------------------------------

    def _compute_fluxes(self):
        '''
        Computes momentum and potential temperature fluxes, and their zonal averages.
        '''
        self._logger.print('computing fluxes and flux zonal means...')
        self._upvp         = self._up * self._vp
        self._upvp.name    = 'upvp'
        self._upvpb        = self._zonal_mean(self._upvp)
        self._upvpb.name   = 'upvpb'
        self._upwapp       = self._up * self._wapp
        self._upwapp.name  = 'upwapp'
        self._upwappb      = self._zonal_mean(self._upwapp)
        self._upwappb.name = 'upwappb'
        self._vptp         = self._vp * self._thetap
        self._vptp.name    = 'vptp'
        self._vptpb        = self._zonal_mean(self._vptp)
        self._vptpb.name   = 'vptpb'
        self._logger.print('computing tracer fluxes and tracer flux zonal means...')
        self._qpvp, self.qpvpb     = [None]*self.ntrac, [None]*welf.ntrac
        self._qpwapp, self.qpwappb = [None]*self.ntrac, [None]*welf.ntrac
        for i in range(len(self.ntrac)): 
            self._qpvp[i]       = self._qp[i] * self._vp
            self._qpvp[i].name  = 'qpvp'
            self._qpvpb[i]      = self._zonal_mean(self._qpvp[i])
            self._qpvpb[i].name = 'qpvpb'
            self._qpwap[i]       = self._qp[i] * self._wap
            self._qpwap[i].name  = 'qpwap'
            self._qpwapb[i]      = self._zonal_mean(self._qpwap[i])
            self._qpwapb[i].name = 'qpwapb'
    
    # --------------------------------------------------
    
    def _compute_derivatives(self):
        '''
        Computes vertical and meridional derivatives, and their zonal averages.
        '''
        self._logger.print('computing psi, vertical derivatives, meridional derivatives...')
        self._dub_dp          = self._p_gradient(self._ub, self.p, self._logger)
        self._dthetab_dp      = self._p_gradient(self._thetab, self.p, self._logger)

        self._ubcoslat        = self._multiply_lat(self._ub, self.coslat)
        self._dubcoslat_dlat  = lat_gradient(self._ubcoslat, np.deg2rad(self.lat))
        
        # ψ = bar(v'* θ') / (dθ'/dp)
        self._psi                  = self._vptpb / self._dthetab_dp 
        self._psicoslat            = self._multiply_lat(self._psi, self.coslat)
        self._dpsicoslat_dlat      = lat_gradient(self._psicoslat, np.deg2rad(self.lat))
        self._dpsi_dp              = self._p_gradient(self._psi, self.p, self._logger) 
       
        self._int_vbdp  = self._p_integral(self._vb, self.p, self._logger)
        
        # tracers
        self._dqb_dp          = self._p_gradient(self._qb, self.p, self._logger)
        self._qbcoslat        = self._multiply_lat(self._qb, self.coslat)
        self._dqbcoslat_dlat  = lat_gradient(self._qbcoslat, np.deg2rad(self.lat))

    # --------------------------------------------------

    def _vtem(self):
        '''
        Returns the TEM northward wind in m/s.
        '''
        # bar(v)* = bar(v) - dψ/dp
        return self._vb - self._dpsi_dp 
    
    # --------------------------------------------------

    def _omegatem(self):
        '''
        Returns the TEM upward wind in Pa/s.
        '''
        # bar(ω)* = bar(ω) + 1/(a*cos(φ)) * d(ψ*cos(φ))/dφ
        return self._wapb + self._multiply_lat(self._dpsicoslat_dlat, 1/(a*self.coslat))
    
    # --------------------------------------------------

    def _wtem(self):
        '''
        Returns the TEM upward wind in m/s.
        '''
        # bar(ω)* = bar(ω) + 1/(a*cos(φ)) * d(ψ*cos(φ))/dφ
        # -> bar(w)* = -H/p * bar(ω)*
        return self._multiply_pres(self._omegatem(), -H/self.p)
    
    # --------------------------------------------------

    def _psitem(self):
        '''
        Returns the TEM mass stream function in kg/s.
        '''
        # Ψ(p) = 2π*a*cos(φ)/g * (int_p^0[bar(v)dp] - ψ)
        return 2*pi*a/g0 * self._multiply_lat(self._int_vbdp - self._psi, self.coslat)
    
    # --------------------------------------------------
    
    def _epfy(self):
        '''
        Returns the northward component of the EP flux in m3/s2, in log-pressure coordinate
        '''
        # ^F_Φ = p/p0 * ( a*cos(φ) * (d(bar(u))/dp * ψ - bar(u'*v') ))
        x = self._multiply_lat(self._dub_dp * self._psi - self._upvpb, a*self.coslat)
        return self._multiply_pres(x, self.p/self.p0)
    
    # --------------------------------------------------

    def _epfz(self):
        '''
        Returns the upward component of the EP flux in m3/s2, in log-pressure coordinate
        ''' 
        if(self._ptype == 'var'): f = self.f
        else: f = self.f[:, np.newaxis, np.newaxis]

        # ^F_z = -H/p0 * a*cos(φ) * (( f - 1/(a*cos(φ)) * d(bar(u)*cos(φ))/dφ )*ψ - bar(u'*ω'))
        x = f - self._multiply_lat(self._dubcoslat_dlat, 1/(a*self.coslat))
        return -H/self.p0 * self._multiply_lat((x*self._psi - self._upwappb), a*self.coslat)
    
    # --------------------------------------------------

    def _epdiv(self):
        '''
        Returns the EP flux divergence.
        '''
        # ∇ * F = 1/(a * cos(φ)) * d(F_φ*cos(φ))/dφ + d(F_p)/dp
        Fphi = self._epfy()
        Fp   = self._epfz() * -self.p0/H
       
        Fphicoslat       = self._multiply_lat(Fphi, self.coslat)
        dFphicoslat_dlat = lat_gradient(Fphicoslat, np.deg2rad(self.lat))
        dFp_dp           = self._p_gradient(Fp, self.p)
        
        return self._multiply_lat(dFphicoslat_dlat, 1/(a*self.coslat)) + dFp_dp
  
    # --------------------------------------------------
 
    def _utendepfd(self):
        '''
        Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        '''
        # d(bar(u))/dt|_(∇ * F) = (∇ * F) / (a*cos(φ))
        return self._multiply_lat(self._epdiv(), 1/(a * self.coslat))

    
    # --------------------------------------------------
 
    def _utendvtem(self):
        '''
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
        '''
        if(self._ptype == 'var'): f = self.f
        else: f = self.f[:, np.newaxis, np.newaxis]
        
        # d(bar(u))/dt|_adv(bar(v)*) = bar(v)* * (f - 1/(a*cos(φ)) * d(bar(u)cos(φ))/dφ)
        vstar = self._vtem()
        diff = (f - self._multiply_lat(self._dubcoslat_dlat, 1/(a*self.coslat)))
        return vstar * diff
    
    # --------------------------------------------------

    def _utendwtem(self):
        '''
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        ''' 
        # d(bar(u))/dt|_adv(bar(ω)*) = -bar(ω)* * d(bar(u)/dp
        wstar = self._omegatem()
        return -wstar * self._dub_dp 

    # --------------------------------------------------
    
    def _etfy(self):
        '''
        Returns the northward component of the eddy tracer flux in m2/s, in log-pressure 
        coordinate
        '''
        # ^M_Φ = p/p0 * ( a*cos(φ) * (d(bar(q))/dp * ψ - bar(q'*v') ))
        x = self._multiply_lat(self._dqb_dp * self._psi - self._qpvpb, a*self.coslat)
        return self._multiply_pres(x, self.p/self.p0)
    
    # --------------------------------------------------

    def _etfz(self):
        '''
        Returns the upward component of the eddy tracer flux in m2/s, in log-pressure 
        coordinate
        ''' 
        if(self._ptype == 'var'): f = self.f
        else: f = self.f[:, np.newaxis, np.newaxis]

        # ^M_z = -H/p0 * a*cos(φ) * ( -1/(a*cos(φ)) * d(bar(q)*cos(φ))/dφ )*ψ - bar(q'*ω'))
        x = -self._multiply_lat(self._dqbcoslat_dlat, 1/(a*self.coslat))
        return -H/self.p0 * self._multiply_lat((x*self._psi - self._qpwappb), a*self.coslat)
    
    # --------------------------------------------------

    def _etdiv(self):
        '''
        Returns the EP flux divergence.
        '''
        # ∇ * M = 1/(a * cos(φ)) * d(M_φ*cos(φ))/dφ + d(F_p)/dp
        # the functions etfy and etfz compute the log-pressure versions of the 
        # vector components, while the present calculation requires the pressure
        # versions. Cancel the conversion factors first
        Mphi = self.multiply_pres(self._etfy(), self.p0/self.p)
        Mp   = self._eptz() * -self.p0/H
       
        Mphicoslat       = self._multiply_lat(Mphi, self.coslat)
        dMphicoslat_dlat = lat_gradient(Mphicoslat, np.deg2rad(self.lat))
        dMp_dp           = self._p_gradient(Mp, self.p)
        
        return self._multiply_lat(dMphicoslat_dlat, 1/(a*self.coslat)) + dMp_dp
  
    # --------------------------------------------------
 
    def _qtendetfd(self):
        '''
        Returns the tendency of tracer mixing ratios due to eddy tracer flux 
        divergence in m/s2.
        '''
        # d(bar(q))/dt|_(∇ * M) = (∇ * M) / (a*cos(φ))
        return self._multiply_lat(self._etdiv(), 1/(a * self.coslat))

    
    # --------------------------------------------------
 
    def _qtendvtem(self):
        '''
        Returns the tendency of tracer mixing ratios due to TEM northward wind 
        advection in m/s2.
        '''
        if(self._ptype == 'var'): f = self.f
        else: f = self.f[:, np.newaxis, np.newaxis]
        
        # d(bar(q))/dt|_adv(bar(v)*) = -bar(v)* * (1/(a*cos(φ)) * d(bar(q)cos(φ))/dφ)
        vstar = self._vtem()
        diff  = self._multiply_lat(self._dqbcoslat_dlat, 1/(a*self.coslat)))
        return -vstar * diff
    
    # --------------------------------------------------

    def _qtendwtem(self):
        '''
        Returns the tendency of tracer mixing ratio due to TEM upward wind 
        advection in m/s2.
        ''' 
        # d(bar(q))/dt|_adv(bar(ω)*) = -bar(ω)* * d(bar(q)/dp
        wstar = self._omegatem()
        return -wstar * self._dqb_dp 

    # --------------------------------------------------

    def to_netcdf(self, loc=os.getcwd(), prefix=None, include_attrs=False):
        '''
        Saves all TEM quantities computed by this class to a NetCDF file.

        Parameters
        ----------
        loc : str, optional
            Location to sace the file. Defaults to the current working directory
        includ_attrs : bool, optional
            Whether or not to write out all of the attributes of this object. 
            Defaults to false, in which case only the quantities computed by
            the class methods are written.
        ''' 
         
        attrs = {"ub":self.ub , "up":self.up, "vb":self.vb , "vp":self.vp,
                 "thetab":self.thetab , "thetap":self.thetap, "wapb":self.wapb , 
                 "wawpp":self.wapp, "upvp":self.upvp , "upvpb":self.upvpb, 
                 "upwapp":self.upwapp , "upwappb":self.upwappb,"vptp":self.vptp , 
                 "vptpb":self.vptpb, "qpvp":self.qpvp, "qpwapp":self.qpwapp, 
                 "qpvpb":self.qpvpb, "qpwappb":self.qpwappb,
                 "dub_dp":self.dub_dp, "dthetab_dp":self.dthetab_dp,
                 "ubcoslat":self.ubcoslat, "dubcoslat_dlat":self.dubcoslat_dlat, 
                 "psi":self.psi, "psicoslat":self.psicoslat,
                 "dpsicoslat_dlat":self.dpsicoslat_dlat, "dpsi_dp":self.dpsi_dp, 
                 "int_vbdp":self.int_vbdp, "dqp_dp":self.dqb_dp, 
                 "qbcoslat":self.qbcoslat, "dqbcoslat_dlat":self.dqbcoslat_dlat}
        results = {"vtem":self.vtem(), "omegatem":self.omegatem(), "wtem":self.wtem(), 
                   "psitem":self.psitem(), 
                   "epfy":self.epfy() , "epfz":self.epfz(), "epdiv":self.epdiv()
                   "utendepfd":self.utendepfd() , 
                   "utendvtem":self.utendvtem(), "utendwtem":self.utendwtem(), 
                   "etfy":self.etfy(), "etfz":self.etfz(), "etdiv":self.etdiv(), 
                   "qtendetfd":self.qtendetfd(), 
                   "qtendvtem":self.qtendvtem(), "qtendwtem":self.qtendwtem()}
        if(include_attrs):
            output = dict(attrs, **results)
        else:
            output = results

        if prefix is not None: prefix = '{}_'.format(prefix)
        else: prefix = ''
        
        filename       = '{}TEM_{}_{}_p{}_L{}_poles{}_attrs{}.nc'.format(prefix,
                         self.ZM.grid_name, self.ZM.grid_out_name, self._ptype, self.L, 
                         self.zm_pole_points, include_attrs)
        self._out_file = '{}/{}'.format(loc, filename)

        dataset = xr.Dataset(output)
        dataset.to_netcdf(self._out_file)
        self._logger.print('wrote TEM data to {}'.format(self._out_file))
        

# =========================================================================


