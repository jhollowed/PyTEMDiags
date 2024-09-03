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
DEFAULT_DIMS = {'horz':'ncol', 'vert':'plev', 'time':'time'}


# -------------------------------------------------------------------------


class TEMDiagnostics:
    def __init__(self, ua, va, ta, wap, p, lat_native, q=None, p0=P0, 
                 zm_dlat=1, L=150, dim_names=DEFAULT_DIMS, 
                 grid_name=None, zm_grid_name=None, map_save_dest=None, 
                 overwrite_map=False, zm_pole_points=False, debug_level=1):
        '''
        This class provides interfaces for computing TEM diagnostic quantities on a pressure 
        vertical coordinate. Upon initialization, the input data is checked and reshaped as 
        necessary, zonal-averaging matrices are built, and zonal mean and eddy components are 
        computed. The available data inputs and outputs (and their naming conventions) are 
        derived from the DynVarMIP experimental protocol (Gerber+Manzini 2016). Funcitonality 
        consistent with these conventions is also included for computing the tracer TEM terms 
        as described in Abalos+ 2017.

        Parameters
        ----------
        ua : xarray DataArray
            Eastward wind, in m/s. This and all other data inputs are expected in at least 2D
            (unstructured spatially 3D), with optional third temporal dimension. The 
            horizontal, vertical, and optional temporal dimensions must be named, with names
            matching those given by the argument 'dim_names'. The expected units of the 
            coordinates are: ncol dimensionless, lev in hPa, time in hours. For an accurate
            computation, all variables should be provided on temporal resolution equal to
            or better than daily.
        va : xarray DataArray
            Northward wind, in m/s.
        ta : xarray DataArray
            Air temperature, in K.
        wap : xarray DataArray
            Vertical pressure velocity, in Pa/s.
        p : xarray DataArray
            Air pressure, in Pa as a 1D array, matching the length of the data variables in 
            the vertical dimension
        lat_native : xarray DataArray
            Latitudes in degrees.
        q : xarray DataArray, or list of xarray DataArray
            dimensionless tracer mass mixing ratios (e.g. kg/kg). This arugment can support any
            arbitrary number of tracer species. If passed as a list, then the list elements 
            should each be xarray DataArrays, each corresponding to a unique tracer. Dimenionality
            and units of the arrays should match the description of the other variables.
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
        debug_level : int, optional
            Integer controlling the printing of verbose progress statements to stdout.
            0 = print no debug statements (defualt)
            1 = print debug statements from PyTEMDiags
            2 = print debug statements from PyTEMDiags and sph_zonal_mean
        
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
        q_out_file : str
            Output filename, used for naming tracer data files written by the q_to_netcdf() method. 
            Simple string that contains metadata about the current TEM option configuration.
        '''

        self._logger = util.logger(debug_level>0, header=True)
         
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
        self.lat_native  = lat_native   # latitudes [deg]
        # options
        self.L              = L
        self.zm_dlat        = zm_dlat
        self.dim_names      = dim_names
        self.zm_pole_points = zm_pole_points
        self.grid_name      = grid_name
        self.zm_grid_name   = zm_grid_name
        self.map_save_dest  = map_save_dest
        self.overwrite_map  = overwrite_map
        self.debug_level    = debug_level

        self.nozflip = nozflip # tmp debug
          
        # ---- veryify input data dimensions, configure data and settings
        self._config_dims()
        
        # ---- construct zonal averaging obeject
        self._logger.print('Getting zonal averaging matrices...')
        self.ZM = sph_zonal_averager(self.lat_native, self._lat_zm, self.L,  
                                     grid_name=grid_name, grid_out_name=zm_grid_name, 
                                     save_dest=map_save_dest, debug=debug_level>1, 
                                     overwrite=overwrite_map)
        if(self.ZM.Y0 is None or self.ZM.Y0p is None):
            self.ZM.sph_compute_matrices(overwrite=overwrite_map)
        self._zonal_mean = self.ZM.sph_zonal_mean # shorthand
        
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
        self.plevname    = self.dim_names['vert']
        # use default time dim name if not present in data
        try: self.timename   = self.dim_names['time']
        except KeyError: self.timename = DEFAULT_DIMS['time']
        self.data_dims = (self.ncolname, self.plevname, self.timename)

        # ---- ensure tracers are a list of DataArrays
        bad_q = False
        if(self.q is not None):
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
            self.ntrac = len(self.q) # record number of input tracers
        else:
            self.ntrac = 0
        self._logger.print('Number of input tracers: {}'.format(self.ntrac))
        # output filename used by self.q_to_netcdf()
        self._q_out_file = [None]*self.ntrac

        # ---- gather variables for dimensions config 
        allvars    = {'ua':self.ua, 'va':self.va, 'ta':self.ta, 
                      'wap':self.wap, 'lat':self.lat_native}
        if(self.ntrac > 0):
            alltracers = dict(('q{}'.format(i), self.q[i]) for i in range(len(self.q)))
            allvars    = {**allvars, **alltracers}
 
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
                                                                 self.plevname, self.timename))
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
        for i in range(self.ntrac):
            old_dims = self.q[i].dims
            self.q[i] = self.q[i].transpose(*self.data_dims)
            self._logger.print('Variable q[{}] transposed: {} -> {}'.format(i, old_dims, self.q[i].dims))
         
        # ---- get coordinates, data lengths
        self.plev  = self.ua[self.plevname]  # model levels [hPa]
        self.time = self.ua[self.timename] # time positions [hours]
        self.NCOL = self.ua.shape[0]
        self.NLEV = self.ua.shape[1]
        self.NT   = self.ua.shape[2] 
        self._logger.print('DATA DIMENSIONS: {} x {} x {} = {} x {} x {}'.format(
                                                   self.ncolname, self.plevname, self.timename, 
                                                   self.NCOL, self.NLEV, self.NT))
        
        # ---- ensure pressure was input as a 1D coordinate consistent with the data
        if not (len(self.p) == self.NLEV and len(self.p.dims) == 1 and self.p.dims[0] == self.plevname):
            raise RuntimeError('pressure p must be input as a 1D coordinate matching '\
                               'the vertical dimension of the data in length (currently {})'\
                               'and name (currently {})'.format(self.NLEV, self.plevname))
        
        # ---- check pressure direction convention
        # ensure that pressure increases toward right-end of arrays. If not, flip
        # this axis for all data
        if(self.plev[0] > self.plev[-1]):
            self.ua  = self.ua.reindex({self.plevname:self.plev[::-1]})
            self.va = self.va.reindex({self.plevname:self.plev[::-1]})
            self.ta  = self.ta.reindex({self.plevname:self.plev[::-1]})
            self.wap = self.wap.reindex({self.plevname:self.plev[::-1]})
            self.p   = self.p.reindex({self.plevname:self.plev[::-1]})
            for i in range(self.ntrac):
                self.q[i] = self.q[i].reindex({self.plevname:self.plev[::-1]}) 
            self.plev = self.ua[self.plevname]
            self._logger.print('Reversed direction of vertical dimension for all data '\
                               '(such that the model top is the leftmost entry in the '\
                               'pressure data array)')
        
        # ---- build uniform latitude discretization lat_zm for zonal mean remap
        tol = 1e-6                  # tolerance for float coparisons
        assert (180/self.zm_dlat).is_integer(), '180 must be divisible by dlat_out'
        
        self._lat_zm = np.arange(-90, 90+self.zm_dlat, self.zm_dlat)
        if(self._lat_zm[-1] > 90+tol): self._lat_zm = self._lat_zm[:-1]
        if(not self.zm_pole_points):
            self._lat_zm = (self._lat_zm[1:] + self._lat_zm[:-1]) / 2
        
        self.ZM_N = len(self._lat_zm)   # length of zonal mean latitude set
        self._logger.print('Built zonal-mean latitude grid with {} deg spacing: {}'.format(
                                                                self.zm_dlat, self._lat_zm))
        
        # --- get latitude-based quantities
        self._f_zm      = 2*Om*np.sin(self._lat_zm * np.pi/180) # coriolis parameter on zonal mean grid
        self._coslat_zm = np.cos(self._lat_zm * np.pi/180)      # cosine of latitude on zonal mean grid 
        # shorthand for quantities used in TEM calculations
        self.lat, self.coslat  = self._lat_zm, self._coslat_zm
        self.f                 = self._f_zm[:, np.newaxis, np.newaxis]
  
    # --------------------------------------------------
    
    '''
    Getter functions for zonally-averaged attributes.
    '''
    @property
    def ub(self): return self._ub
    @property
    def vb(self): return self._vb
    @property
    def thetab(self): return self._thetab
    @property
    def wapb(self): return self._wapb
    @property
    def up(self): return self._up
    @property
    def vp(self): return self._vp
    @property
    def thetap(self): return self._thetap
    @property
    def wapp(self): return self._wapp
    @property
    def upvp(self): return self._upvp
    @property
    def upwapp(self): return self._upwapp
    @property
    def vptp(self): return self._vptp
    @property
    def upvpb(self): return self._upvpb
    @property
    def upwappb(self): return self._upwappb
    @property
    def vptpb(self): return self._vptpb
    @property
    def dub_dp(self): return self._dub_dp
    @property
    def dthetab_dp(self): return self._dthetab_dp
    @property
    def ubcoslat(self): return self._ubcoslat
    @property
    def dubcoslat_dlat(self): return self._dubcoslat_dlat
    @property
    def psicoslat(self): return self._psicoslat
    @property
    def dpsicoslat_dlat(self): return self._dpsicoslat_dlat
    @property
    def int_vbdp(self): return self._int_vbdp
    @property
    def psi(self): return self._psi
    @property
    def dpsi_dp(self): return self._dpsi_dp
    @property
    def qp(self): return self._qp
    @property
    def qpvp(self): return self._qpvp
    @property
    def qpwapp(self): return self._qpwapp    
    @property
    def qb(self): return self._qb
    @property
    def qpvpb(self): return self._qpvpb
    @property
    def qpwappb(self): return self._qpwappb
    @property
    def dqb_dp(self): return self._dqb_dp
    @property
    def qbcoslat(self): return self._qbcoslat
    @property
    def dqbcoslat_dlat(self): return self._dqbcoslat_dlat
    @property
    def out_file(self): 
        if(self._out_file is None):
            warnings.warn('\'out_file\' is not set until to_netcdf() is called')
        return self._out_file
    @property
    def q_out_file(self):
        if(len(self._q_out_file) == 0):
            warnings.warn('\'q_out_file\' is emtpy; no tracers currently present')
        if(self._q_out_file.count(None) == self.ntrac):
            warnings.warn('\'q_out_file\' is not set until q_to_netcdf() is called')
        return self._q_out_file

    # --------------------------------------------------
    
    def _compute_potential_temperature(self):
        '''
        Computes the potential temperature from temperature and pressure.
        '''
        self._logger.print('computing potential temperature...')

        # θ = T * (p0/p)**k
        theta = util.multiply_p(self.ta, (self.p0/self.p)**k)
       
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
        self._qb, self._qp = [None]*self.ntrac, [None]*self.ntrac
        for i in range(self.ntrac): 
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
        
        self._qpvp, self._qpvpb     = [None]*self.ntrac, [None]*self.ntrac
        self._qpwapp, self._qpwappb = [None]*self.ntrac, [None]*self.ntrac
        for i in range(self.ntrac): 
            self._qpvp[i]         = self._qp[i] * self._vp
            self._qpvp[i].name    = 'qpvp'
            self._qpvpb[i]        = self._zonal_mean(self._qpvp[i])
            self._qpvpb[i].name   = 'qpvpb'
            self._qpwapp[i]       = self._qp[i] * self._wapp
            self._qpwapp[i].name  = 'qpwapp'
            self._qpwappb[i]      = self._zonal_mean(self._qpwapp[i])
            self._qpwappb[i].name = 'qpwappb'
    
    # --------------------------------------------------
    
    def _compute_derivatives(self):
        '''
        Computes vertical and meridional derivatives, and their zonal averages.
        '''
        self._logger.print('computing psi, vertical derivatives, meridional derivatives...')
        self._dub_dp          = util.p_gradient(self._ub, self.p)
        self._dub_dp.name     ='dub_dp'
        self._dthetab_dp      = util.p_gradient(self._thetab, self.p)
        self._dthetab_dp.name = 'dthetab_dp'

        self._ubcoslat        = util.multiply_lat(self._ub, self.coslat)
        self._ubcoslat.name   = 'ubcoslat'
        self._dubcoslat_dlat  = util.lat_gradient(self._ubcoslat, np.deg2rad(self.lat))
        self._dubcoslat       = 'dubcoslat'
        
        # ψ = bar(v'* θ') / (dθ'/dp)
        self._psi                  = self._vptpb / self._dthetab_dp
        self._psi.name             = 'psi'
        self._psicoslat            = util.multiply_lat(self._psi, self.coslat)
        self._psicoslat.name       = 'psicoslat'
        self._dpsicoslat_dlat      = util.lat_gradient(self._psicoslat, np.deg2rad(self.lat))
        self._dpsicoslat_dlat.name = 'dpsicoslat_dlat'
        self._dpsi_dp              = util.p_gradient(self._psi, self.p) 
        self._dpsi_dp.name         = 'dpsi_dp'
       
        self._int_vbdp      = util.p_integral(self._vb, self.p)
        self._int_vbdp.name = 'int_vbdp'
        
        self._dqb_dp         = [None]*self.ntrac
        self._qbcoslat       = [None]*self.ntrac
        self._dqbcoslat_dlat = [None]*self.ntrac
        for i in range(self.ntrac):
            self._dqb_dp[i]              = util.p_gradient(self._qb[i], self.p)
            self._dqb_dp[i].name         = 'dqb_dp'
            self._qbcoslat[i]            = util.multiply_lat(self._qb[i], self.coslat)
            self._qbcoslat[i].name       = 'dqbcoslat'
            self._dqbcoslat_dlat[i]      = util.lat_gradient(self._qbcoslat[i], np.deg2rad(self.lat))
            self._dqbcoslat_dlat[i].name = 'dqbcoslat_dlat'

    # --------------------------------------------------

    def vtem(self):
        '''
        Returns the TEM northward wind in m/s.
        '''
        self._logger.print('computing vtem...')

        # bar(v)* = bar(v) - dψ/dp
        vtem = self._vb - self._dpsi_dp
        vtem.name = 'vtem'

        # match data type to type of input data
        vtem = vtem.astype(self.va.dtype)

        return vtem
    
    # --------------------------------------------------

    def omegatem(self):
        '''
        Returns the TEM upward wind in Pa/s.
        '''
        self._logger.print('computing omegatem...')
        
        # bar(ω)* = bar(ω) + 1/(a*cos(φ)) * d(ψ*cos(φ))/dφ
        omegatem = self._wapb + util.multiply_lat(self._dpsicoslat_dlat, 1/(a*self.coslat))
        omegatem.name = 'omegatem'
        
        # match data type to type of input data
        omegatem = omegatem.astype(self.wap.dtype)
        
        return omegatem
    
    # --------------------------------------------------

    def wtem(self):
        '''
        Returns the TEM upward wind in m/s.
        '''
        self._logger.print('computing wtem...')
        
        # bar(ω)* = bar(ω) + 1/(a*cos(φ)) * d(ψ*cos(φ))/dφ
        # -> bar(w)* = -H/p * bar(ω)*
        wtem = util.multiply_p(self.omegatem(), -H/self.p)
        wtem.name = 'wtem'
        
        # match data type to type of input data
        wtem = wtem.astype(self.wap.dtype)
        
        return wtem
    
    # --------------------------------------------------

    def psitem(self):
        '''
        Returns the TEM mass stream function in kg/s.
        '''
        self._logger.print('computing psitem...')
        
        # Ψ(p) = 2π*a*cos(φ)/g * (int_p^0[bar(v)dp] - ψ)
        psitem = 2*pi*a/g0 * util.multiply_lat(self._int_vbdp - self._psi, self.coslat)
        psitem.name = 'psitem'
        
        # match data type to type of input data
        psitem = psitem.astype(self.va.dtype)
        
        return psitem
    
    # --------------------------------------------------
    
    def epfy(self):
        '''
        Returns the northward component of the EP flux in m3/s2, in log-pressure coordinate
        '''
        self._logger.print('computing epfy...')
        
        # ^F_Φ = p/p0 * ( a*cos(φ) * (d(bar(u))/dp * ψ - bar(u'*v') ))
        x    = util.multiply_lat(self._dub_dp * self._psi - self._upvpb, a*self.coslat)
        epfy = util.multiply_p(x, self.p/self.p0)
        epfy.name = 'epfy'
        
        # match data type to type of input data
        epfy = epfy.astype(self.ua.dtype)
        
        return epfy
    
    # --------------------------------------------------

    def epfz(self):
        '''
        Returns the upward component of the EP flux in m3/s2, in log-pressure coordinate
        ''' 
        self._logger.print('computing epfz...')

        # ^F_z = -H/p0 * a*cos(φ) * (( f - 1/(a*cos(φ)) * d(bar(u)*cos(φ))/dφ )*ψ - bar(u'*ω'))
        x    = self.f - util.multiply_lat(self._dubcoslat_dlat, 1/(a*self.coslat))
        epfz = -H/self.p0 * util.multiply_lat((x*self._psi - self._upwappb), a*self.coslat)
        epfz.name = 'epfz'
        
        # match data type to type of input data
        epfz = epfz.astype(self.ua.dtype)
        
        return epfz
    
    # --------------------------------------------------

    def epdiv(self):
        '''
        Returns the EP flux divergence.
        '''
        self._logger.print('computing epdiv...')
        
        # ∇ * F = 1/(a * cos(φ)) * d(F_φ*cos(φ))/dφ + d(F_p)/dp
        # the functions epfy and epfz compute the log-pressure versions of the 
        # vector components, while the present calculation requires the pressure
        # versions. Cancel the conversion factors first
        Fphi = util.multiply_p(self.epfy(), self.p0/self.p)
        Fp   = self.epfz() * -self.p0/H
       
        Fphicoslat       = util.multiply_lat(Fphi, self.coslat)
        dFphicoslat_dlat = util.lat_gradient(Fphicoslat, np.deg2rad(self.lat))
        dFp_dp           = util.p_gradient(Fp, self.p)
        epdiv            = util.multiply_lat(dFphicoslat_dlat, 1/(a*self.coslat)) + dFp_dp
        epdiv.name = 'epdiv' 
        
        # match data type to type of input data
        epdiv = epdiv.astype(self.ua.dtype)
        
        return epdiv
  
    # --------------------------------------------------
 
    def utendepfd(self):
        '''
        Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        '''
        self._logger.print('computing utendepfd...')
        
        # d(bar(u))/dt|_(∇ * F) = (∇ * F) / (a*cos(φ))
        utendepfd = util.multiply_lat(self.epdiv(), 1/(a * self.coslat))
        utendepfd.name = 'utendepfd'
        
        # match data type to type of input data
        utendepfd = utendepfd.astype(self.ua.dtype)
        
        return utendepfd
    
    # --------------------------------------------------
 
    def utendvtem(self):
        '''
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
        '''
        self._logger.print('computing utendvtem...')
         
        # d(bar(u))/dt|_adv(bar(v)*) = bar(v)* * (f - 1/(a*cos(φ)) * d(bar(u)cos(φ))/dφ)
        vstar     = self.vtem()
        diff      = (self.f - util.multiply_lat(self._dubcoslat_dlat, 1/(a*self.coslat)))
        utendvtem = vstar * diff
        utendvtem.name = 'utendvtem'
        
        # match data type to type of input data
        utendvtem = utendvtem.astype(self.ua.dtype)
        
        return utendvtem
    
    # --------------------------------------------------

    def utendwtem(self):
        '''
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        ''' 
        self._logger.print('computing utendwtem...')

        # d(bar(u))/dt|_adv(bar(ω)*) = -bar(ω)* * d(bar(u)/dp
        wstar     = self.omegatem()
        utendwtem = -wstar * self._dub_dp
        utendwtem.name = 'utendwtem'
        
        # match data type to type of input data
        utendwtem = utendwtem.astype(self.ua.dtype)
        
        return utendwtem

    # --------------------------------------------------
    
    def etfy(self, qi=None):
        '''
        Returns the northward component of the eddy tracer flux in m2/s, in log-pressure 
        coordinate

        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing etfy...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
                  'qi must be passed to etfy() when len(q) > 1!')

        psi = self._psi
        dqb_dp = self._dqb_dp[qi]
        qpvpb = self._qpvpb[qi]

        # ^M_Φ = p/p0 * ( a*cos(φ) * (d(bar(q))/dp * ψ - bar(q'*v') ))
        x    = util.multiply_lat(dqb_dp * psi - qpvpb, a*self.coslat)
        etfy = util.multiply_p(x, self.p/self.p0)
        etfy.name = 'etfy'
        
        # match data type to type of input data
        etfy = etfy.astype(self.q[qi].dtype)
        
        return etfy
    
    # --------------------------------------------------

    def etfz(self, qi=None):
        '''
        Returns the upward component of the eddy tracer flux in m2/s, in log-pressure 
        coordinate
        
        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing etfz...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
                  'qi must be passed to etfz() when len(q) > 1!')
        
        psi = self._psi
        dqbcoslat_dlat = self._dqbcoslat_dlat[qi]
        qpwappb = self._qpwappb[qi]

        # ^M_z = -H/p0 * a*cos(φ) * ( -1/(a*cos(φ)) * d(bar(q)*cos(φ))/dφ )*ψ - bar(q'*ω'))
        x    = -util.multiply_lat(dqbcoslat_dlat, 1/(a*self.coslat))
        etfz = -H/self.p0 * util.multiply_lat((x*psi - qpwappb), a*self.coslat)
        etfz.name = 'etfz'
        
        # match data type to type of input data
        etfz = etfz.astype(self.q[qi].dtype)
        
        return etfz
    
    # --------------------------------------------------

    def etdiv(self, qi=None):
        '''
        Returns the EP flux divergence.
        
        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing etdiv...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
                 'qi must be passed to etdiv() when len(q) > 1!')
        
        # ∇ * M = 1/(a * cos(φ)) * d(M_φ*cos(φ))/dφ + d(F_p)/dp
        # the functions etfy and etfz compute the log-pressure versions of the 
        # vector components, while the present calculation requires the pressure
        # versions. Cancel the conversion factors first
        Mphi = util.multiply_p(self.etfy(qi), self.p0/self.p)
        Mp   = self.etfz(qi) * -self.p0/H
       
        Mphicoslat       = util.multiply_lat(Mphi, self.coslat)
        dMphicoslat_dlat = util.lat_gradient(Mphicoslat, np.deg2rad(self.lat))
        dMp_dp           = util.p_gradient(Mp, self.p)
        etdiv            = util.multiply_lat(dMphicoslat_dlat, 1/(a*self.coslat)) + dMp_dp
        etdiv.name = 'etdiv' 
        
        # match data type to type of input data
        etdiv = etdiv.astype(self.q[qi].dtype)
        
        return etdiv
  
    # --------------------------------------------------
 
    def qtendetfd(self, qi=None):
        '''
        Returns the tendency of tracer mixing ratios due to eddy tracer flux 
        divergence in m/s2.
        
        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing qtendetfd...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
             'qi must be passed to qtendetfd() when len(q) > 1!')
        
        # d(bar(q))/dt|_(∇ * M) = (∇ * M) / (a*cos(φ))
        qtendetfd = util.multiply_lat(self.etdiv(qi), 1/(a * self.coslat))
        qtendetfd.name = 'qtendetfd'
        
        # match data type to type of input data
        qtendetfd = qtendetfd.astype(self.q[qi].dtype)
        
        return qtendetfd
 
    # --------------------------------------------------
 
    def qtendvtem(self, qi=None):
        '''
        Returns the tendency of tracer mixing ratios due to TEM northward wind 
        advection in m/s2.
        
        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing qtendvtem...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
             'qi must be passed to qtendvtem() when len(q) > 1!')
        
        vstar = self.vtem()
        dqbcoslat_dlat = self._dqbcoslat_dlat[qi]
        
        # d(bar(q))/dt|_adv(bar(v)*) = -bar(v)* * (1/(a*cos(φ)) * d(bar(q)cos(φ))/dφ)
        diff      = util.multiply_lat(dqbcoslat_dlat, 1/(a*self.coslat))
        qtendvtem = -vstar * diff
        qtendvtem.name = 'qtendvtem'
        
        # match data type to type of input data
        qtendvtem = qtendvtem.astype(self.q[qi].dtype)
        
        return qtendvtem
    
    # --------------------------------------------------

    def qtendwtem(self, qi=None):
        '''
        Returns the tendency of tracer mixing ratio due to TEM upward wind 
        advection in m/s2.
        
        Parameters
        ----------
        qi : int, optional
            Tracer index to use for computation. qi=0 will return the result for
            the tracer at q[0]. Must be passed if q includes more than one tracer.
        '''
        self._logger.print('computing qtendwtem...')
        
        if(qi is None and self.ntrac == 1): qi = 0
        elif(qi is None and self.ntrac > 1): raise RuntimeError(
             'qi must be passed to qtendwtem() when len(q) > 1!')
        
        wstar = self.wtem()
        dqb_dp = self._dqb_dp[qi]
        
        # d(bar(q))/dt|_adv(bar(ω)*) = -bar(ω)* * d(bar(q)/dp
        wstar     = self.omegatem()
        qtendwtem = -wstar * dqb_dp
        qtendwtem.name = 'qtendwtem'
        
        # match data type to type of input data
        qtendwtem = qtendwtem.astype(self.q[qi].dtype)
        
        return qtendwtem

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
         
        attrs   = {"ub":self.ub , "up":self.up, "vb":self.vb , "vp":self.vp,
                   "thetab":self.thetab , "thetap":self.thetap, "wapb":self.wapb , 
                   "wawpp":self.wapp, "upvp":self.upvp , "upvpb":self.upvpb, 
                   "upwapp":self.upwapp , "upwappb":self.upwappb,"vptp":self.vptp , 
                   "vptpb":self.vptpb, "dub_dp":self.dub_dp, "dthetab_dp":self.dthetab_dp,
                   "ubcoslat":self.ubcoslat, "dubcoslat_dlat":self.dubcoslat_dlat, 
                   "psi":self.psi, "psicoslat":self.psicoslat,
                   "dpsicoslat_dlat":self.dpsicoslat_dlat, "dpsi_dp":self.dpsi_dp, 
                   "int_vbdp":self.int_vbdp}
        results = {"vtem":self.vtem(), "omegatem":self.omegatem(), "wtem":self.wtem(), 
                   "psitem":self.psitem(), 
                   "epfy":self.epfy(), "epfz":self.epfz(), "epdiv":self.epdiv(),
                   "utendepfd":self.utendepfd(), 
                   "utendvtem":self.utendvtem(), "utendwtem":self.utendwtem()}
        if(include_attrs):
            output = dict(attrs, **results)
        else:
            output = results

        if prefix is not None: prefix = '{}_'.format(prefix)
        else: prefix = ''
        
        filename       = '{}TEM_{}_{}_L{}.nc'.format(prefix,
                         self.ZM.grid_name, self.ZM.grid_out_name, self.L)
        #filename       = '{}TEM_{}_{}_L{}_poles{}_attrs{}.nc'.format(prefix,
        #                 self.ZM.grid_name, self.ZM.grid_out_name, self.L, 
        #                 self.zm_pole_points, include_attrs)
        self._out_file = '{}/{}'.format(loc, filename)

        dataset = xr.Dataset(output)
        dataset.to_netcdf(self._out_file)
        self._logger.print('wrote TEM data to {}'.format(self._out_file))
        return self._out_file
    
    # --------------------------------------------------

    def q_to_netcdf(self, loc=os.getcwd(), qi=None, prefix=None, include_attrs=False):
        '''
        Saves all TEM tracer quantities computed by this class to NetCDF files. Tracer
        names will be included in the output file name, which are deduced from the input
        tracer DataArrays q. If these DataArrays do not have names, they will be assigned
        generaically as "q1", "q2", ...

        Parameters
        ----------
        qi : int, optional;
            Index of tracer to write out. If not povided, all tracer TEM quantities
            will be written out to separate files per-tracer
        loc : str, optional
            Location to sace the file. Defaults to the current working directory
        includ_attrs : bool, optional
            Whether or not to write out all of the tracer attributes of this object. 
            Defaults to false, in which case only the quantities computed by
            the class methods are written.
        '''

        assert self.ntrac > 0, 'No tracers to output (argument `q` not passed at object construction)'
        
        if prefix is not None: prefix = '{}_'.format(prefix)
        else: prefix = ''

        # get tracer names
        tracer_names = [qii.name for qii in self.q]
        for i in range(self.ntrac): 
            if(tracer_names[i] is None): tracer_names[i] = 'q{}'.format(i)

        # get output indices 
        if qi is None: qi = np.arange(self.ntrac)
        else: qi = [qi]
        
        for i in qi:
            attrs   = {"qpvp":self.qpvp[i], "qpwapp":self.qpwapp[i], 
                       "qpvpb":self.qpvpb[i], "qpwappb":self.qpwappb[i], "dqp_dp":self.dqb_dp[i], 
                       "qbcoslat":self.qbcoslat[i], "dqbcoslat_dlat":self.dqbcoslat_dlat[i]}
            results = {"etfy":self.etfy(i), "etfz":self.etfz(i), "etdiv":self.etdiv(i), 
                       "qtendetfd":self.qtendetfd(i), 
                       "qtendvtem":self.qtendvtem(i), "qtendwtem":self.qtendwtem(i)}
            if(include_attrs):
                output = dict(attrs, **results)
            else:
                output = results
 
            filename       = '{}TEM_{}_{}_L{}_TRACER-{}.nc'.format(prefix,
                             self.ZM.grid_name, self.ZM.grid_out_name, self.L, 
                             tracer_names[i])
            #filename       = '{}TEM_{}_{}_L{}_poles{}_attrs{}_TRACER-{}.nc'.format(prefix,
            #                 self.ZM.grid_name, self.ZM.grid_out_name, self.L, 
            #                 self.zm_pole_points, include_attrs, tracer_names[i])
            self._q_out_file[i] = '{}/{}'.format(loc, filename)

            dataset = xr.Dataset(output)
            dataset.to_netcdf(self._q_out_file[i])
            self._logger.print('wrote {} tracer TEM data to {}'.format(
                                       tracer_names[i], self._q_out_file[i]))
        return self._q_out_file
        

# =========================================================================


