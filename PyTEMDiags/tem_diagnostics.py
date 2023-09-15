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

    def __init__(self, ua, va, ta, wap, p, lat, p0=P0, zm_dlat=1, L=150, 
                 dim_names=DEFAULT_DIMS, log_pressure=True, grid_name=None, 
                 zm_grid_name=None, map_save_dest=None, overwrite_map=False, 
                 zm_pole_points=False, debug=False):
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
            Air pressure, in Pa.
        lat : xarray DataArray
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
        ZM : sph_zonal_averager object
            Zonal averager object associated with current configuration.
        zm_lat : 1D array
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
            "u-bar", zonal mean of ua in m/s, matching length of zm_lat in the horizontral, 
            otherwise matching dims of input ua.
        vb : N-D array
            "v-bar", zonal mean of va in m/s, matching length of zm_lat in the horizontral, 
            otherwise matching dims of input va.
        wapb : N-D array
            "omega-bar", zonal mean of wap in Pa/a, matching length of zm_lat in the 
            horizontral, otherwise matching dims of input wap.
        thetab : N-D array
            "theta-bar", zonal mean of theta in K, matching length of zm_lat in the 
            horizontral, otherwise matching dims of input ta.
        pb : N-D array
            "p-bar", zonal mean of pressure in Pa, matching length of zm_lat in the 
            horizontral, otherwise matching dims of input p.
        upvp : N-D array
            Product of up and vp in m2/s2, aka northward flux of eastward momentum, matching 
            dims of input ua.
        upvpb : N-D array
            Zonal mean of upvp in m2/s2, matching length of zm_lat in the horizontal, 
            otherwise matching dims of input ua.
        upwapp : N-D array
            Product of up and wapp in (m Pa)/s2, aka upward flux of eastward momentum, 
            matching dims of input ua.
        upwwappb : N-D array
            Zonal mean of upwapp in (m Pa)/s2, matching length of zm_lat in the horizontal, 
            otherwise matching dims of input ua.
        vptp : N-D array       
            Product of vp and thetap in (m K)/s, aka northward flux of potential temperature,
            matching dims of input va.
        vptpb : N-D array
            Zonal mean of vptp in (m K)/s, matching length of zm_lat in the horizontal, 
            otherwise matching dims of input va.
        f : 1D array
            Coriolis parameter on native grid.
        zm_f : 1D array
            Coriolis parameter on zonal mean grid.
        coslat : 1D array
            Cosine of native grid latitude.
        zm_coslat : 1D array
            Cosine of zonal mean latitude.
        '''

        self.logger = logger(debug, header=True)
        
        # ---- get input args
        # variables
        self.p    = p    # pressure [Pa]
        self.ua   = ua   # eastward wind [m/s]
        self.va   = va   # northward wind [m/s]
        self.ta   = ta   # temperature [K]
        self.wap  = wap  # vertical pressure velocity [Pa/s]
        self.lat  = lat  # latitudes [deg]
        self.p0   = p0   # reference pressure [Pa]
        # options
        self.L              = L
        self.debug          = debug
        self.zm_dlat        = zm_dlat
        self.log_pres       = log_pressure
        self.dim_names      = dim_names
        self.zm_pole_points = zm_pole_points
        
        # ---- declare new fields
        self.zm_lat    = None # zonal mean latitude set [deg]
        self.theta     = None # potential temperature [K]
        self.up        = None # "u-prime", zonally asymmetric anomaly in ua [m/s]
        self.vp        = None # "v-prime", zonally asymmetric anomaly in va [m/s]
        self.wapp      = None # "wap-prime", zonally asymmetric anomaly in wap [Pa/s]
        self.thetap    = None # "theta-prime", zonally asymmetric anomaly in theta [K]
        self.ub        = None # "u-bar", zonal mean of ua [m/s]
        self.vb        = None # "v-bar", zonal mean of va [m/s]
        self.wapb      = None # "wap-bar", zonal mean of wap [Pa/s]
        self.thetab    = None # "theta-bar", zonal mean of theta [K]
        self.upvp      = None # product of up and vp, aka northward flux of eastward 
                              # momentum [m2/s2]
        self.upvpb     = None # zonal mean of upvp [m2/s2]
        self.upwapp    = None # product of up and wapp, aka upward flux of eastward 
                              # momentum [(m Pa)/s2]
        self.upwwappb  = None # zonal mean of upwapp [(m Pa)/s2]
        self.vptp      = None # product of vp and thetap, aka northward flux of potential 
                              # temperature [(m K)/s]
        self.vptpb     = None # zonal mean of vptp [(m K)/s]
         
        # ---- handle dimensions of input data, generate latitude-based quantities
        self._handle_dims()
        self.f         = 2*Om*np.sin(self.lat)    # coriolis parameter on native grid
        self.zm_f      = 2*Om*np.sin(self.zm_lat) # coriolis parameter on zonal mean grid
        self.coslat    = np.cos(self.lat)         # cosine of latitude
        self.zm_coslat = np.cos(self.zm_lat)      # cosine of latitude
       
        # ---- get potential temperature
        self._compute_potential_temperature()
        
        # ---- construct zonal averaging obeject, get zonal means, eddy components
        self.logger.print('Getting zonal averaging matrices...')
        self.ZM = sph_zonal_averager(self.lat, self.zm_lat, self.L, 
                                     grid_name, zm_grid_name, map_save_dest, debug=debug)
        if(self.ZM.Z is None or self.ZM.Zp is None):
            self.ZM.sph_compute_matrices()
        self._decompose_zm_eddy()
        
        # ---- compute fluxes, vertical and meridional derivaties, streamfunction terms
        self._compute_fluxes()
        self._compute_derivatives()


    # --------------------------------------------------


    def _handle_dims(self):
        '''
        Checks that input data dimensions are as expected. Adds temporal dimensions if needed.
        Transposes input data such that data shapes are consistent with (ncol, lev, time). Builds
        latitude discretization for zonal means.
        '''

        # ---- get dim names
        self.ncoldim   = self.dim_names['horz']
        self.levdim    = self.dim_names['vert']
        # use default time dim name if not present in data
        try: self.timedim   = self.dim_names['time']
        except KeyError: self.timedim = DEFAULT_DIMS['time']
        self.data_dims = (self.ncoldim, self.levdim, self.timedim)
                
        allvars     = {'ua':self.ua, 'va':self.va, 'ta':self.ta, 
                       'wap':self.wap, 'p':self.p, 'lat':self.lat}
 
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
            if(dat.shape[dat.dims.index(self.ncoldim)] != len(self.lat)):
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
                self.logger.print('Expanded variable {} to contain new dimension \'{}\''.format(
                                                                              var, self.timedim))
        # ---- reshape data as (ncol, lev, time)
        # data is now verified to meet the expected input criteria; 
        # next transpose data to standard format with ncol as the first dimension
        # (this must be done outside of the loop above to avoid allocating memory 
        #  for copies of the underlying data objects in the dict allvars, I think)
        old_dims = self.ua.dims
        self.ua = self.ua.transpose(*self.data_dims)
        self.logger.print('Variable ua transposed: {} -> {}'.format(old_dims, self.ua.dims))
        old_dims = self.va.dims
        self.va = self.va.transpose(*self.data_dims)
        self.logger.print('Variable va transposed: {} -> {}'.format(old_dims, self.va.dims))
        old_dims = self.ta.dims
        self.ta = self.ta.transpose(*self.data_dims)
        self.logger.print('Variable ta transposed: {} -> {}'.format(old_dims, self.ta.dims))
        old_dims = self.wap.dims
        self.wap = self.wap.transpose(*self.data_dims)
        self.logger.print('Variable wap transposed: {} -> {}'.format(old_dims, self.wap.dims))
        old_dims = self.p.dims
        self.p = self.p.transpose(*self.data_dims)
        self.logger.print('Variable p transposed: {} -> {}'.format(old_dims, self.p.dims))
        
        # ---- get coordinates, data lengths
        self.lev  = self.ua[self.levdim]   # model levels [hPa]
        self.time = self.ua[self.timedim]  # time positions [hours]
        self.NCOL = self.ua.shape[0]
        self.NLEV = self.ua.shape[1]  
        self.NT   = self.ua.shape[2] 
        self.logger.print('DATA DIMENSIONS: {} x {} x {} = {} x {} x {}'.format(
                                                   self.ncoldim, self.levdim, self.timedim, 
                                                   self.NCOL, self.NLEV, self.NT))
        
        # ---- build uniform latitude discretization zm_lat for zonal mean remap
        tol = 1e-6                  # tolerance for float coparisons
        assert (180/self.zm_dlat).is_integer(), '180 must be divisible by dlat_out'
        
        self.zm_lat = np.arange(-90, 90+self.zm_dlat, self.zm_dlat)
        if(self.zm_lat[-1] > 90+tol): self.zm_lat = self.zm_lat[:-1]
        if(not self.zm_pole_points):
            self.zm_lat = (self.zm_lat[1:] + self.zm_lat[:-1]) / 2
        
        self.ZM_N = len(self.zm_lat)   # length of zonal mean latitude set
        self.logger.print('Built zonal-mean latitude grid with {} deg spacing: {}'.format(
                                                                self.zm_dlat, self.zm_lat))
    

    # --------------------------------------------------
    

    def _compute_potential_temperature(self):
        '''
        Computes the potential temperature from temperature and pressure.
        '''
        self.theta = self.ta * (self.p0/self.p)**k
        self.theta.name = 'THETA'


    # --------------------------------------------------
    

    def _decompose_zm_eddy(self):
        '''
        Decomposes atmospheric variables into zonal means and eddy components.
        ''' 
        self.ub     = self.ZM.sph_zonal_mean_native(self.ua) 
        self.up     = self.ua - self.ub
        self.vb     = self.ZM.sph_zonal_mean_native(self.va) 
        self.vp     = self.va - self.vb
        self.thetab = self.ZM.sph_zonal_mean_native(self.theta) 
        self.thetap = self.theta - self.thetab
        self.wapb   = self.ZM.sph_zonal_mean_native(self.wap) 
        self.wapp   = self.wap - self.wapb
    

    # --------------------------------------------------
    

    def _compute_fluxes(self):
        '''
        Computes momentum and potential temperature fluxes, and their zonal averages.
        '''
        self.upvp    = self.up * self.vp
        self.upvpb   = self.ZM.sph_zonal_mean(self.upvp)
        self.upwapp  = self.up * self.wapp
        self.upwappb = self.ZM.sph_zonal_mean(self.upwapp)
        self.vptp    = self.vp * self.thetap
        self.vptpb   = self.ZM.sph_zonal_mean(self.vptp)
    

    # --------------------------------------------------
    

    def _compute_derivatives(self):
        '''
        Computes vertical and meridional derivatives, and their zonal averages.
        '''
        self.dub_dp          = p_gradient(self.ub_native, self.p)
        self.dthetab_dp      = p_gradient(self.thetab_native, self.p)
 
        pdb.set_trace()
        ubcoslat             = np.einsum('ijk,i->ijk', self.ub, self.coslat)
        self.dubcoslat_dlat  = lat_gradient(ubcoslat, self.lat)
        
        self.psi             = self.vptpb / self.dthetab_dp 
        psicoslat            = np.einsum('ijk,i->ijk', self.psi, self.coslat)
        self.dpsicoslat_dlat = lat_gradient(psicoslat, self.lat)
        self.dpsi_dp         = p_gradient(self.psi, self.pb) 
    

    # --------------------------------------------------
    
    '''
    Below are a set of getter functions for all zonally-averaged quantities.
    Internal to the class, these quantities are defined on the native grid, 
    where all computations occur (this is done since the vertical coordinate 
    is unstructured in pressure; computing everything on the native grid avoids 
    needing to assume model level pressure, or interpolate to pressure levels).
    It would be obnoxious, however, to return to the user zonal averages with 
    NCOL data points. Instead, these getter functions remap the quantities to
    zm_lat.
    '''
    @property
    def ub(self): return self.ZM.sph_zonal_mean(self.ub)
    @property
    def vb(self): return self.ZM.sph_zonal_mean(self.vb)
    @property
    def thetab(self): return self.ZM.sph_zonal_mean(self.thetab)
    @property
    def wapb(self): return self.ZM.sph_zonal_mean(self.wapb)
    @property
    def upvpb(self): return self.ZM.sph_zonal_mean(self.upvpb)
    @property
    def upwappb(self): return self.ZM.sph_zonal_mean(self.upwappb)
    @property
    def vptpb(self): return self.ZM.sph_zonal_mean(self.vptpb)
    @property
    def dub_bp(self): return self.ZM.sph_zonal_mean(self.dub_bp)
    @property
    def dthetab_dp(self): return self.ZM.sph_zonal_mean(self.dthetab_dp)
    @property
    def dubcoslat_dlat(self): return self.ZM.sph_zonal_mean(self.dubcoslat_dlat)
    @property
    def dpsicoslat_dlat(self): return self.ZM.sph_zonal_mean(self.dpsicoslat_dlat)
    @property
    def dpsi_dp(self): return self.ZM.sph_zonal_mean(self.dpsi_dp)

    # --------------------------------------------------


    def momentum_budget(self):
        '''
        Desc
        '''
        pass
    
    # --------------------------------------------------

    # ****FIXED
    def epfy(self):
        '''
        Returns the northward component of the EP flux in m3/s2.
        '''
        #p/p0 * ( a*lat * (du/dp * psi - avg(up*vp) ))
        x = np.einsum('i,ijk->ijk', a*self.coslat, self.dubdp * self.psi - self.dupvpb)
        return np.einsum('j,ijk->ijk', self.p/self.p0, x)
    
    # --------------------------------------------------

    # ****FIXED
    def epfz(self):
        '''
        Returns the upward component of the EP flux in m3s2.
        '''
        x = f - np.einsum('ijk,i->ijk', self.dubcoslat_dlat, 1/(a*self.coslat))
        return -H/self.p0 * np.einsum('i,ijk->ijk', a*self.coslat, (x*self.psi - self.upwappb))
    
    # --------------------------------------------------

    def epdiv(self):
        '''
        Returns the EP flux divergence.
        '''
        Fphi = self.epfy()
        Fp   = self.epfz()
        
        dFphicoslat_dlat = lat_gradient(Fphi*self.zm_coslat, self.zm_lat)
        dFp_dp           = p_gradient(Fp, self.pb)
        
        return dFphicoslat_dlat + dFp_dp
    
    # --------------------------------------------------

    def vtem(self):
        '''
        Returns the TEM northward wind in m/s.
        '''
        return self.vb - self.dpsi_dp 
    
    # --------------------------------------------------

    def wtem(self):
        '''
        Returns the TEM upward wind in m/s.
        '''
        return self.wapb + self.dpsicoslat_dlat / (a*self.zm_coslat)
    
    # --------------------------------------------------

    def psitem(self):
        '''
        Returns the TEM mass stream function in kg/s.
        '''
        intvb_p  = np.trapz(self.vb, self.p, axis=1)
        return 2*pi*a*self.zm_coslat/g0 * self.intvb_pi
    
    # --------------------------------------------------
 
    def utendepfd(self):
        '''
            Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        '''
        return self.epdiv() / (a * self.zm_coslat)

    
    # --------------------------------------------------
 
    def utendvtem(self):
        '''
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
        '''
        vstar = self.vtem()
        return vstar * (self.zm_f - self.dubcoslat_dlat / (a*self.coslat))
    
    # --------------------------------------------------

    def utendwtem(self):
        '''
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        '''
        wstar = self.wtem
        return -wstar * self.dub_dp
        

# =========================================================================


