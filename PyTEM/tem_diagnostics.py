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

    def __init__(self, ua, va, ta, wap, p, lat, zm_dlat=1, L=50, 
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

        self.logger = util.logger(debug)
        
        # ---- get input args
        # variables
        self.p    = p    # pressure [Pa]
        self.ua   = ua   # eastward wind [m/s]
        self.va   = va   # northward wind [m/s]
        self.ta   = ta   # temperature [K]
        self.wap  = wap  # vertical pressure velocity [Pa/s]
        self.lat  = lat  # latitudes [deg]
        # options
        self.L         = L        # max. spherical harmonic degree 
        self.debug     = debug    # logging bool flag
        self.zm_dlat   = zm_dlat  # zonal mean discretization spacing [deg]
        self.log_pres  = log_pres # log pressure bool flag
        
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
        self.upvp      = None # product of up and vp, aka northward flux of eastward momentum [m2/s2]
        self.upvpb     = None # zonal mean of upvp [m2/s2]
        self.upwapp    = None # product of up and wapp, aka upward flux of eastward momentum [(m Pa)/s2]
        self.upwwappb  = None # zonal mean of upwapp [(m Pa)/s2]
        self.vptp      = None # product of vp and thetap, aka northward flux of potential temperature [(m K)/s]
        self.vptpb     = None # zonal mean of vptp [(m K)/s]
        self.f         = 2*Om*np.sin(self.lat)    # coriolis parameter on native grid
        self.zm_f      = 2*Om*np.sin(self.zm_lat) # coriolis parameter on zonal mean grid
        self.coslat    = np.cos(self.lat)         # cosine of latitude
        self.zm_coslat = np.cos(self.zm_lat)      # cosine of latitude
         
        # ---- handle dimensions of input data, generate coordinate arrays
        self._handle_dims()
        
        # ---- get potential temperature
        self._compute_potential_temperature()
        
        # ---- construct zonal averaging obeject, get zonal means, eddy components
        self.logger.print('Getting zonal averaging matrices...')
        self.ZM = sph_zonal_averager(self.lat, self.zm_lat, self.L, 
                                           grid_name, zm_grid_name, map_save_dest)
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
    

    def _compute_potential_temperature():
        '''
        Computes the potential temperature from temperature and pressure.
        '''
        self.theta = self.ta * (p0/self.p)**k
        self.theta.name = 'THETA'


    # --------------------------------------------------
    

    def _decompose_zm_eddy():
        '''
        Decomposes atmospheric variables into zonal means and eddy components.
        ''' 
        self.up      = self.ua - self.ZM.sph_zonal_mean_native(self.ua)
        self.ub      = self.ZM.sph_zonal_mean(self.ua) 
        self.vp      = self.va - self.ZM.sph_zonal_mean_native(self.va)
        self.vb      = self.ZM.sph_zonal_mean(self.va) 
        self.pb      = self.ZM.sph_zonal_mean(self.p) 
        self.thetap  = self.theta - self.ZM.sph_zonal_mean_native(self.theta)
        self.thetab  = self.ZM.sph_zonal_mean(self.theta) 
        self.wapp    = self.wap - self.ZM.sph_zonal_mean_native(self.wap)
        self.wapb    = self.ZM.sph_zonal_mean(self.wap) 
    

    # --------------------------------------------------
    

    def _compute_fluxes():
        '''
        Computes momentum and potential temperature fluxes, and their zonal averages.
        '''
        self.upvp     = self.up * self.vp
        self.upvpb    = self.ZM.sph_zonal_mean(self.upvp)
        self.upwapp   = self.up * self.wapp
        self.upwwappb = self.ZM.sph_zonal_mean(upwapp)
        self.vptp     = self.vp * self.thetap
        self.vptpb    = self.ZM.sph_zonal_mean(self.vptp)
    

    # --------------------------------------------------
    

    def _compute_derivatives():
        '''
        Computes vertical and meridional derivatives, and their zonal averages.
        '''
        self.dub_dp     = np.gradient(self.ub, self.pb, axis=1)
        self.dthetab_dp = np.gradient(self.thetab, self.pb, axis=1)
 
        ubcoslat            = self.ub * self.zm_coslat
        self.dubcoslat_dlat = np.gradient(uVbcoslat, self.zm_lat, axis=0)
        
        self.psi             = self.vptpb / self.dthetab_dp 
        psicoslat            = self.psi * self.zm_coslat
        self.dpsicoslat_dlat = np.gradient(psicoslat, self.zm_lat, axis=0)
        self.dpsi_dp         = np.gradient(self.psi, self.pb, axis=1) 
    

    # --------------------------------------------------


    def momentum_budget():
        '''
        Desc
        '''
        pass
    
    # --------------------------------------------------

    def epfy():
        '''
        Returns the northward component of the EP flux in m3/s2.
        '''
        return self.p/p0 * ( a*self.zm_coslat * (self.dubdp * self.psi - self.dpvpb) )
    
    # --------------------------------------------------

    def epfz():
        '''
        Returns the upward component of the EP flux in m3s2.
        '''
        x = (zm_f - self.dubcoslat_dlat/(a*self.zm_coslat))
        return -H/p0 * (a*self.zm_coslat * (x*self.psi - self.upwappb) )
    
    # --------------------------------------------------

    def epdiv():
        '''
        Returns the EP flux divergence.
        '''
        Fphi = self.epfy()
        Fp   = self.epfz()
        
        dFphicoslat_dlat = np.gradient(Fphi*self.zm_coslat, self.zm_lat, axis=0)
        dFp_dp           = np.gradient(Fp, self.pb, axis=1)
        
        return dFphicoslat_dlat + dFp_dp
    
    # --------------------------------------------------

    def vtem():
        '''
        Returns the TEM northward wind in m/s.
        '''
        return self.vb - self.dpsi_dp 
    
    # --------------------------------------------------

    def wtem():
        '''
        Returns the TEM upward wind in m/s.
        '''
        return self.wapb + self.dpsicoslat_dlat / (a*self.zm_coslat)
    
    # --------------------------------------------------

    def psitem():
        '''
        Returns the TEM mass stream function in kg/s.
        '''
        intvb_p  = np.trapz(self.vb, self.p, axis=1)
        return 2*pi*a*self.zm_coslat/g0 * self.intvb_pi
    
    # --------------------------------------------------
 
    def utendepfd():
        '''
            Returns the tendency of eastward wind due to EP flux divergence in m/s2.
        '''
        return self.epdiv() / (a * self.zm_coslat)

    
    # --------------------------------------------------
 
    def utendvtem():
        '''
        Returns the tendency of eastward wind due to TEM northward wind advection 
        and coriolis in m/s2.
        '''
        vstar = self.vtem()
        return vstar * (self.zm_f - self.dubcoslat_dlat / (a*self.coslat))
    
    # --------------------------------------------------

    def utendwtem():
        '''
        Returns the tendency of eastward wind due to TEM upward wind advection in m/s2.
        '''
        wstar = self.wtem
        return -wstar * self.dub_dp
        

# =========================================================================


