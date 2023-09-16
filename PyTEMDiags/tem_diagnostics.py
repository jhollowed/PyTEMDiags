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
from .tem_util import *
from .constants import *
from .sph_zonal_mean import *

# ---- global constants
DEFAULT_DIMS = {'horz':'ncol', 'vert':'lev', 'time':'time'}

# -------------------------------------------------------------------------


class TEMDiagnostics:
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
    lat_native : xarray DataArray
        Latitudes in degrees.
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
    epfy()
        Returns the northward component of the EP flux in m3/s2.
    epfz()
        Returns the upward component of the EP flux in m3s2.
    epdiv()
        Returns the EP flux divergence.
    vtem()
        Returns the TEM northward wind in m/s.
    wtem()
        Returns the TEM upward wind in m/s.
    psitem()
        Returns the TEM mass stream function in kg/s.
    utendepfd()
        Returns the tendency of eastward wind due to EP flux divergence in m/s2.
    utendvtem()
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
    utendwtem()
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
    
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
    '''
    

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
    def ub(self): return self._zm_return_func(self.ub)
    @ub.setter
    def set_ub(self, x): self.ub = x
    @property
    def vb(self): return self._zm_return_func(self.vb)
    @vb.setter
    def set_vb(self, x): self.vb = x
    @property
    def thetab(self): return self._zm_return_func(self.thetab)
    @thetab.setter
    def set_thetab(self, x): self.thetab = x
    @property
    def wapb(self): return self._zm_return_func(self.wapb)
    @wapb.setter
    def set_wapb(self, x): self.wapb = x
    @property
    def upvpb(self): return self._zm_return_func(self.upvpb)
    @upvpb.setter
    def set_upvpb(self, x): self.upvpb = x
    @property
    def upwappb(self): return self._zm_return_func(self.upwappb)
    @upwappb.setter
    def set_upwappb(self, x): self.upwappb = x
    @property
    def vptpb(self): return self._zm_return_func(self.vptpb)
    @vptpb.setter
    def set_vptpb(self, x): self.vptpb = x
    @property
    def dub_dp(self): return self._zm_return_func(self.dub_dp)
    @dub_dp.setter
    def set_dub_dp(self, x): self.dub_dp = x
    @property
    def dthetab_dp(self): return self._zm_return_func(self.dthetab_dp)
    @dthetab_dp.setter
    def set_dthetab_dp(self, x): self.dthetab_dp = x
    @property
    def dubcoslat_dlat(self): return self._zm_return_func(self.dubcoslat_dlat)
    @dubcoslat_dlat.setter
    def set_dubcoslat_dlat(self, x): self.dubcoslat_dlat = x
    @property
    def dpsicoslat_dlat(self): return self._zm_return_func(self.dpsicoslat_dlat)
    @dpsicoslat_dlat.setter
    def set_dpsicoslat_dlat(self, x): self.dpsicoslat_dlat = x
    @property
    def dpsi_dp(self): return self._zm_return_func(self.dpsi_dp)
    @dpsi_dp.setter
    def set_dpsi_dp(self, x): self.dpsi_dp = x
    

    # --------------------------------------------------


    def __init__(self, ua, va, ta, wap, p, lat_native, p0=P0, zm_dlat=1, L=150, 
                 dim_names=DEFAULT_DIMS, log_pressure=True, grid_name=None, 
                 zm_grid_name=None, map_save_dest=None, overwrite_map=False, 
                 zm_pole_points=False, debug=False):

        self._logger = logger(debug, header=True)
        
        # ---- get input args
        # variables
        self.p    = p    # pressure [Pa]
        self.ua   = ua   # eastward wind [m/s]
        self.va   = va   # northward wind [m/s]
        self.ta   = ta   # temperature [K]
        self.wap  = wap  # vertical pressure velocity [Pa/s]
        self.p0   = p0   # reference pressure [Pa]
        self.lat_native = lat_native  # latitudes [deg]
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
                                     grid_name, zm_grid_name, map_save_dest, debug=debug)
        if(self.ZM.Z is None or self.ZM.Zp is None):
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
     
    
    # --------------------------------------------------


    def _config_dims(self):
        '''
        Checks that input data dimensions are as expected. Adds temporal dimensions 
        if needed. Transposes input data such that data shapes are consistent with 
        (ncol, lev, time). Builds latitude discretization for zonal means.
        '''

        # ---- get dim names
        self.ncoldim   = self.dim_names['horz']
        self.levdim    = self.dim_names['vert']
        # use default time dim name if not present in data
        try: self.timedim   = self.dim_names['time']
        except KeyError: self.timedim = DEFAULT_DIMS['time']
        self.data_dims = (self.ncoldim, self.levdim, self.timedim)
        
        allvars     = {'ua':self.ua, 'va':self.va, 'ta':self.ta, 
                       'wap':self.wap, 'lat':self.lat_native}

        # ---- check if pressure was input as a coordinate (1D) or a variable (>1D)
        if len(self.p.dims) == 1 and self.p.dims[0] == self.levdim:
            self._ptype = 'coord'
        elif self.p.dims == self.ua.dims and self.levdim in self.p.dims:
            self._ptype = 'var'
            allvars['p'] = self.p
        else:
            raise RuntimeError('pressure p must be input as either a 1D coordinate matching '\
                               'the vertical dimension of the data in length (currently {}), '\
                               'or a variable with dimensionality matching the data '\
                               '(currently {})'.format(
                                self.ua.shape[self.ua.dims.index(self.levdim)], self.ua.dims))
            
        # ---- verify dat input format, transform as needed
        for var,dat in allvars.items():
            if(not isinstance(dat, xr.core.dataarray.DataArray)):
                raise RuntimeError('Input data for arg \'{}\' must be an '\
                                   'xarray DataArray'.format(var))
            # check data dimensions
            if(self.ncoldim not in dat.dims):
                raise RuntimeError('Input data {} does not contain dim {}'.format(
                                                                var, self.ncoldim)) 
            # ensure passed lat and data ncol are consistent
            if(dat.shape[dat.dims.index(self.ncoldim)] != len(self.lat_native)):
                raise RuntimeError('Dimension {} in variable {} is length {}, but input '\
                                   'parameter lat is length {}; these must match!'.format(
                                           self.ncoldim, var, var.shape[0], len(self.lat)))
            if(var == 'lat'): continue
            # input must have 2 or 3 dimensions 
            if(len(dat.dims) < 2 or len(dat.dims) > 3):
                raise RuntimeError('Input data has {0} dims, expected either 2 ({1}, {2}) '\
                                   'or 3 ({1}, {2}, {3})'.format(len(dat.dims), self.ncoldim, 
                                                                 self.levdim, self.timedim))
            # if time doesn't exist, expand the data arrays to include a new temporal
            # dimension of length 1, value 0.
            if(self.timedim not in dat.dims):
                dat = dat.expand_dims(self.timedim, axis=len(dat.dims))
                self._logger.print('Expanded variable {} to contain new '\
                                  'dimension \'{}\''.format(var, self.timedim))

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
        
        # ---- get coordinates, data lengths
        self.lev  = self.ua[self.levdim]   # model levels [hPa]
        self.time = self.ua[self.timedim]  # time positions [hours]
        self.NCOL = self.ua.shape[0]
        self.NLEV = self.ua.shape[1]  
        self.NT   = self.ua.shape[2] 
        self._logger.print('DATA DIMENSIONS: {} x {} x {} = {} x {} x {}'.format(
                                                   self.ncoldim, self.levdim, self.timedim, 
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
        self._f_native = 2*Om*np.sin(self.lat_native) # coriolis parameter on native grid
        self._f_zm     = 2*Om*np.sin(self.lat_zm)     # coriolis parameter on zonal mean grid
        self._coslat_native = np.cos(self.lat_native) # cosine of latitude
        self._coslat_zm     = np.cos(self.lat_zm)     # cosine of latitude

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
            self._logger.print('reduced pressure p from a 3D {} variable to a 1D (length {}) '\
                              'coordinate'.format(self.ua.shape, len(self.p)))
    

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
            self._p_gradient       = p_gradient_1d                 # (a)
            self._p_integral       = p_integral_1d                 # (a)
            self._zonal_mean       = self.ZM.sph_zonal_mean        # (b)
            self.lat, self.coslat  = self.lat_zm, self._coslat_zm  # (c)
            self.f                 = self._f_zm                    # (c)
            self._zm_return_func   = lambda x: x                   # (d)
            self._multiply_pres    = lambda a,p: np.einsum('ijk,j->ijk', a, p) # (e)
            self._logger.print('configuration set for pressure p as a coordinate')
        elif(self._ptype == 'var'):
            self._p_gradient       = p_gradient_3d                        # (f)
            self._p_integral       = p_integral_3d                        # (f)
            self._zonal_mean       = self.ZM.sph_zonal_mean_native        # (g)
            self.lat, self.coslat  = self.lat_native, self._coslat_native # (h)
            self.f                 = self._f_native                       # (h)
            self._zm_return_func   = self.ZM.sph_zonal_mean               # (i)
            self._multiply_pres    = lambda a,p=self.p: a*p               # (j)
            self._logger.print('configuration set for pressure p as a variable')
        self._multiply_lat = lambda a,lat: np.einsum('ijk,i->ijk', a, lat) #(k)


    # --------------------------------------------------
    

    def _compute_potential_temperature(self):
        '''
        Computes the potential temperature from temperature and pressure.
        '''
        # θ = T * (p0/p)**k
        self.theta = self._multiply_pres(self.ta, (self.p0/self.p)**k)
        self.theta.name = 'THETA'


    # --------------------------------------------------
    

    def _decompose_zm_eddy(self):
        '''
        Decomposes atmospheric variables into zonal means and eddy components.
        ''' 
        self.ub     = self._zonal_mean(self.ua) 
        self.up     = self.ua - self.ub
        self.vb     = self._zonal_mean(self.va) 
        self.vp     = self.va - self.vb
        self.thetab = self._zonal_mean(self.theta) 
        self.thetap = self.theta - self.thetab
        self.wapb   = self._zonal_mean(self.wap) 
        self.wapp   = self.wap - self.wapb
    

    # --------------------------------------------------
    

    def _compute_fluxes(self):
        '''
        Computes momentum and potential temperature fluxes, and their zonal averages.
        '''
        self.upvp    = self.up * self.vp
        self.upvpb   = self._zonal_mean(self.upvp)
        self.upwapp  = self.up * self.wapp
        self.upwappb = self._zonal_mean(self.upwapp)
        self.vptp    = self.vp * self.thetap
        self.vptpb   = self._zonal_mean(self.vptp)
    

    # --------------------------------------------------
    

    def _compute_derivatives(self):
        '''
        Computes vertical and meridional derivatives, and their zonal averages.
        '''
        self.dub_dp          = self._p_gradient(self.ub, self.p)
        self.dthetab_dp      = self._p_gradient(self.thetab, self.p)
 
        ubcoslat             = self._multiply_coslatb(self.ub)
        self.dubcoslat_dlat  = lat_gradient(ubcoslat, self.lat)
        
        # ψ = bar(v'* θ') / (dθ'/dp)
        self.psi             = self.vptpb / self.dthetab_dp 
        psicoslat            = self._multiply_coslatb(self.psi)
        self.dpsicoslat_dlat = lat_gradient(psicoslat, self.coslat)
        self.dpsi_dp         = self._p_gradient(self.psi) 
    

    # --------------------------------------------------


    def epfy(self):
        '''
        Returns the northward component of the EP flux in m3/s2.
        '''
        # F_Φ = p/p0 * ( a*cos(φ) * (d(bar(u))/dp * ψ - bar(u'*v') ))
        x = self._multiply_lat(self.dubdp * self.psi - self.dupvpb, a*self.coslat)
        return self._multiply_pres(x, self.p/self.p0)
    
    # --------------------------------------------------

    def epfz(self):
        '''
        Returns the upward component of the EP flux in m3/s2.
        '''
        # F_z = -H/p0 * a*cos(φ) * (( f - 1/(a*cos(φ)) * d(bar(u)*cos(φ))/dφ )*ψ - bar(u'*ω'))
        x = f - self._multiply_lat(self.dubcoslat_dlat, 1/(a*self.coslat))
        return -H/self.p0 * self._multiply_lat((x*self.psi - self.upwappb), a*self.coslat)
    
    # --------------------------------------------------

    def epdiv(self):
        '''
        Returns the EP flux divergence.
        '''
        # ∇ * F = 1/(a * cos(φ)) * d(F_φ*cos(φ))/dφ + d(F_p)/dp
        Fphi = self.epfy()
        Fp   = -self.p0/H * self.epfz()
       
        Fphicoslat       = self._multiply_lat(Fphi, self.coslat)
        dFphicoslat_dlat = lat_gradient(Fphicoslat, self.lat)
        dFp_dp           = self_.p_gradient(Fp, self.p)
        
        return self._multiply_lat(dFphicoslat_dlat, 1/(a*self.coslat)) + dFp_dp
    
    # --------------------------------------------------

    def vtem(self):
        '''
        Returns the TEM northward wind in m/s.
        '''
        # bar(v)* = bar(v) - dψ/dp
        return self.vb - self.dpsi_dp 
    
    # --------------------------------------------------

    def wtem(self):
        '''
        Returns the TEM upward wind in m/s.
        '''
        # bar(ω)* = bar(ω) + 1/(a*cos(φ)) * d(ψ*cos(φ))/dφ
        return self.wapb + self._multiply_lat(self.dpsicoslat_dlat, 1/(a*self.coslat))
    
    # --------------------------------------------------

    def psitem(self):
        '''
        Returns the TEM mass stream function in kg/s.
        '''
        # Ψ(p) = 2π*a*cos(φ)/g * (int_p^0[bar(v)dp] - ψ)
        int_vbdp  = self._p_integrate(self.vb, self.p)
        return 2*pi*a/g0 * self._multiply_lat(int_vbdp - self.psi, self.zm_coslat)
    
    # --------------------------------------------------
 
    def utendepfd(self):
        '''
        Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        '''
        # d(bar(u))/dt|_(∇ * F) = (∇ * F) / (a*cos(φ))
        return self._multiply_lat(self.epdiv(), 1/(a * self.coslat))

    
    # --------------------------------------------------
 
    def utendvtem(self):
        '''
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
        '''
        # d(bar(u))/dt|_adv(bar(v)*) = bar(v)* * (f - 1/(a*cos(φ)) * d(bar(u)cos(φ))/dφ)
        vstar = self.vtem()
        diff = (self.zm_f - self._multiply_lat(self.dubcoslat_dlat, 1/(a*self.coslat)))
        return vstar * diff
    
    # --------------------------------------------------

    def utendwtem(self):
        '''
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        ''' 
        # d(bar(u))/dt|_adv(bar(ω)*) = -bar(ω)* * d(bar(u)/dp
        wstar = self.wtem
        return -wstar * self.dub_dp
        

# =========================================================================


