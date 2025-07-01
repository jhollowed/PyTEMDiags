# Joe Hollowed
# University of Michigan 2024
#
# Providing functions for performing 2D fourier transforms in zonal wavenumber and
# time for input variables. Based on code from the CASanalysis repository from 
# Isla Simpson, found at:
# https://github.com/islasimpson/CASanalysis


# =========================================================================


# ---- import dependencies
import pdb
import numpy as np
import xarray as xr

# ---- imports from this package
from .tem_util import *
from .constants import *

# ---- global constants
SAVE_DEST = '{}/../maps'.format(pathlib.Path(__file__).parent.resolve())
NCENC = {'_FillValue':None}


# -------------------------------------------------------------------------

class wave_spectrum_1d:
    def __init__(self, x, time_bounds=None, lat_band=None):
        '''
        This class provides methods for taking transforms, inverse transforms, and
        spectral filtering on input datasets. Code in several of the functions here
        is inherited from the module filter_utils.py in the CASanalysis repo, found here:
        https://github.com/islasimpson/CASanalysis

        Parameters
        ----------
        x : xarray DataArray
            The data to transform. Dimensions must include 'time', and any other dimensions
        time_bounds : list of float
            List of length 2, giving the left and right bounds of the time window, in days. 
            The subset of data lying outside of this window will be discarded. Default is 
            None, in which the full time domain of the data is used.
        lat_band : float, optional
            List of length 2, giving the southern and northern bounds of the latitude band, 
            in degrees. The subset of data lying outside of this band will be discarded. 
            Default is None, in which the full latitude domain of the data is used.
        
        Public Methods
        -------------
        transform()
            Takes the 1D Foruier transform in time
        invert()
            Takes the inverse transform of the fft() result
        filter()
            Filters the transformed data by frequency bounds in fourier space
        invert_filtered()
            Inverts the filtered transformed data back to real space

        Public Attributes
        -----------------
        self.xfft
            The Fourier transform of the data. Initialized by calling self.transform()
        self.xinv
            The inverse Fourier transform of the data. Initialized by calling self.invert()
        self.frequency
            The frequency coordinate of the transformed data. Initialized by calling self.transform()
        '''

        # verify arguments
        assert(isinstance(x, xarray.core.dataarray.DataArray)), 'x must be an xarray DataArray'

        # verify data dimensions
        self.dims = np.array(x.dims)
        assert('time' in self.dims), 'Data must have a \'time\' dimension'
        if(lat_band is not None):
            assert('lat' in self.dims), 'Data must have a \'lat\' dimension'

        # do data slicing
        if(time_bounds is not None):
            assert(time_bounds[0] > time_bounds[1]), 'time_bounds must be increasing'
            x = x.sel(time=slice(time_bounds[0], time_bounds[1]))
        if(lat_band is not None):
            assert(lat_band[0] > lat_band[1]), 'lat_bounds must be increasing'
            x = x.sel(lat=slice(lat_band[0], lat_band[1]))
        self.x = x

        # get coords
        self.coords  = x.coords
        self.time    = x.time
        self.timedim = x.dims.index('time')
        self.ntime   = self.time.size

        # get data output frequencies
        self.spd = self.time.diff('time').values.astype('timedelta64[D]').astype(int)
        assert(len(np.unique(self.spd)) == 1): 'Data must be uniformly spaced in time!'\
                                               'Data time diff is: {}'.format(self.spd)
        # set empty attributes for methods
        self.xfft = None
        self.xinv = None
        self.frequency = None
                
    # --------------------------------------------------

    def transform(self, ftaper=None):
        '''
        Takes the 1D Fourier transform of the data. The resulting Fourier coefficients are 
        multiplied by two, and the negative frequencies are discarded.

        Parameters
        ----------
        ftaper : float, optional
            The fraction of the data that's used for tapering the data in time.
            ftaper/2 will be tapered at each end. Default is 0.1.
        '''
        
        # arguments
        if(ftaper is None): ftaper = 0.1
        self.ftaper = ftaper
        
        # taper the time series
        npts     = int(np.rint(ftaper * self.ntime))
        endtaper = np.hanning(npts)
        taper    = np.ones(self.ntime)
        taper[0:npts//2+1] = endtaper[0:npts//2+1]
        taper[-npts//2+1:] = endtaper[npts//2+1]
        taper = xr.DataArray(taper, coords=[self.time], dims=['time'], name='taper')
        self.xtapered = self.x * taper

        # do fft in time
        self.xfft = np.fft.fft(self.xfft, axis=self.timedim) / self.ntime

        # assign dimensions are coordinates to fft result
        dimsfft   = []
        coordsfft = []
        for i in range(len(self.dims)):
            if(dims[i] == 'time'):
                dimsfft.append('frequency')
                self.frequency = xr.DataArray(np.fft.fftfreq(self.ntime, 1/self.spd), 
                                                             dims=['frequency']
                coordsfft.append(self.frequency)
            else:
                dimsfft.append(dims[i])
                coordsfft.append(self.coords[dims[i]])
        self.xfft = xr.DataArray(self.xfft, dims=dimsfft, coords=coordsfft)
        return self.xfft
    

    # --------------------------------------------------


    def invert(self, xfft=None):
        '''
        Applies an inverse transform to the transformed data to obtain the data in real space.

        Parameters
        ----------
        xfft : xarray DataArray, optional
            The tranformed data to invert. Default is None, in which case the data is self.xfft,
            which results from self.transform()

        Return
        ------
        xinv : xarray DataArray
            The inverted, filtered data
        '''

        
        if(xfft is None):
            xfft = self.xfft
        
        # invert the filtered data
        xinv = np.fft.ifft(xfft, axis=self.timedim) * xfft.frequency.size
        xinv = 2 * np.real(xinv)
        xinv = xr.DataArray(xinv, dims=dims, coords=coords)

        if(xfft == self.xfft):
            self.xinv = xinv
        return xinv


    # --------------------------------------------------


    def filter(pmin, pmax, xfft=None):
        '''
        Filters the transformed data in fourier space, defined by lower and upper bounds 
        in frequency. The arguments (pmin, pmax) specify the filter bounds in frequency.
        
        Parameters
        ----------
        pmin : int, optional
            Minimum period (in days) included in the output. Default is None.
        pmax : int, optional
            Maximum period (in days) included in the output. Default is None.
        xfft : xarray DataArray, optional
            The tranformed data to filter. Default is None, in which case the data is self.xfft,
            which results from self.transform()

        Returns
        -------
        xfilt : xarray DataArray
            The filtered data in fourier space
        '''

        # check arguments
        if(xfft is None):
            xfft = self.xfft
        assert(pmin>=0 and pmax>=0), 'Requested period range (pmin, pmax) must be positive'
        assert(pmin < pmax), 'pmin must be less than pmax'

        # convert min/max period to min/max freq
        self.fmin, self.fmax = 1/pmax, /pmin
        
        # ensure fft has been run
        assert(self.xfft is not None): 'transform() must be run before filtering'

        # select desired wavenumbers and frequencies
        xfilt = xfft.where(((xfft.frequency >= fmin) & (xfft.frequency <= fmax)), 0)
        return xfilt

    # --------------------------------------------------

    def invert_filtered(pmin, pmax):
        '''
        Filters the transformed data in fourier space, defined by
        lower and upper bounds in frequency, and inverts the result
        back to real space. The arguments (pmin, pmax) specify the filter bounds in 
        frequency.

        Parameters
        ----------
        pmin : int
            Minimum period (in days) included in the output
        pmax : int
            Maximum period (in days) included in the output

        Return
        ------
        xinv : xarray DataArray
            The inverted, filtered data
        '''

        # filter the data
        xfilt = self.filter(pmin, pmax)
        # invert the filtered data
        xinv = self.invert(xfilt)
        return xinv


# ==============================================================================
# ==============================================================================


class wave_spectrum_2d:
    def __init__(self, x, time_bounds=None, lat_band=None):
        '''
        This class provides methods for taking transforms, inverse transforms, and
        spectral filtering on input datasets on a specified time bounds and latitude
        band. Code in several of the functions here is inherited from the module 
        filter_utils.py in the CASanalysis repo, found here:
        https://github.com/islasimpson/CASanalysis

        Parameters
        ----------
        x : xarray DataArray
            The data to transform. Dimensions must include 'time' and 'lon', and any other 
            dimensions
        time_bounds : list of float
            List of length 2, giving the left and right bounds of the time window, in days. 
            The subset of data lying outside of this window will be discarded. Default is 
            None, in which the full time domain of the data is used.
        lat_band : float, optional
            List of length 2, giving the southern and northern bounds of the latitude band, 
            in degrees. The subset of data lying outside of this band will be discarded. 
            Default is None, in which the full latitude domain of the data is used.
        
        Public Methods
        -------------
        transform()
            Takes the 2D foruier transform in lon and time
        invert()
            Takes the inverse transform of the fft() result
        filter()
            Filters the transformed data by frequency and wavenumber bounds in fourier space
        invert_filtered()
            Inverts the filtered transformed data back to real space
        filter_westerly()
            Filters the transformed data for positive wavenumbers (westerly waves)
        filter_easterly()
            Filters the transformed data for negative wavenumbers (easterly waves)
        invert_westerly_filtered()
            Inverts the westerly filterd transform back to real space
        invert_easterly_filtered()
            Inverts the easterly filtered transform back to real space

        Public Attributes
        -----------------
        self.xfft
            The Fourier transform of the data. Initialized by calling self.transform()
        self.xinv
            The inverse Fourier transform of the data. Initialized by calling self.invert()
        self.frequency
            The frequency coordinate of the transformed data. Initialized by calling self.transform()
        self.wavenumber
            The wavenumber coordinate of the transformed data. Initialized by calling self.transform()
        '''

        # verify arguments
        assert(isinstance(x, xarray.core.dataarray.DataArray)), 'x must be an xarray DataArray'

        # verify data dimensions
        self.dims = np.array(x.dims)
        assert('time' in self.dims), 'Data must have a \'time\' dimension'
        assert('lon' in self.dims), 'Data must have a \'lon\' dimension'
        if(lat_band is not None):
            assert('lat' in self.dims), 'Data must have a \'lat\' dimension'

        # do data slicing
        if(time_bounds is not None):
            assert(time_bounds[0] > time_bounds[1]), 'time_bounds must be increasing'
            x = x.sel(time=slice(time_bounds[0], time_bounds[1]))
        if(lat_band is not None):
            assert(lat_bounds[0] > lat_bounds[1]), 'lat_bounds must be increasing'
            x = x.sel(lat=slice(lat_band[0], lat_band[1]))
        self.x = x

        # get coords
        self.coords  = x.coords
        self.time    = x.time
        self.lon     = x.lon
        self.londim  = x.dims.index('lon')
        self.timedim = x.dims.index('time')
        self.nlon    = self.lon.size
        self.ntime   = self.time.size

        # get data output frequencies
        self.spd = self.time.diff('time').values.astype('timedelta64[D]').astype(int)
        assert(len(np.unique(self.spd)) == 1): 'Data must be uniformly spaced in time!'\
                                               'Data time diff is: {}'.format(self.spd)
        self.spd = seldf.spd[0]

        # set empty attributes for methods
        self.xfft = None
        self.xinv = None
        self.frequency  = None
        self.wavenumber = None
            
    # --------------------------------------------------

    def transform(self, ftaper=None):
        '''
        Takes the 2D Fourier transform of the data. In the returned transform, negative wavenumbers
        indicate westward propegating waves, while positive wavenumbers indicate eastward propogating
        waves. The resulting Fourier coefficients are multiplied by two, and the negative frequencies 
        are discarded.

        Parameters
        ----------
        ftaper : float, optional
            The fraction of the data that's used for tapering the data in time.
            ftaper/2 will be tapered at each end. Default is 0.1.
        '''
        
        # arguments
        if(ftaper is None): ftaper = 0.1
        self.ftaper = ftaper
        
        # taper the time series
        npts     = int(np.rint(ftaper * self.ntime))
        endtaper = np.hanning(npts)
        taper    = np.ones(self.ntime)
        taper[0:npts//2+1] = endtaper[0:npts//2+1]
        taper[-npts//2+1:] = endtaper[npts//2+1]
        taper = xr.DataArray(taper, coords=[self.time], dims=['time'], name='taper')
        self.xtapered = self.x * taper

        # do fft in longitude
        self.xfft = np.fft.fft(self.xtapered, axis=self.londim) / self.nlon
        # do fft in time
        self.xfft = np.fft.fft(self.xfft, axis=self.timedim) / self.ntime

        # assign dimensions are coordinates to fft result
        dimsfft   = []
        coordsfft = []
        for i in range(len(self.dims)):
            if(dims[i] == 'time'):
                dimsfft.append('frequency')
                self.frequency = xr.DataArray(np.fft.fftfreq(self.ntime, 1/self.spd), 
                                                             dims=['frequency']
                coordsfft.append(self.frequency)
            elif(dims[i] == 'lon'):
                dimsfft.append('wafvenumber')
                self.wavevnumber = xr.DataArray(-1*np.fft.fftfreq(self.nlon, 1/self.nlon), 
                                                dims=['wavenumber'])
                coordsfft.append(self.wavenumber)
            else:
                dimsfft.append(dims[i])
                coordsfft.append(self.coords[dims[i]])
        self.xfft = xr.DataArray(self.xfft, dims=dimsfft, coords=coordsfft)
    
    # --------------------------------------------------

    def invert(self, xfft=None):
        '''
        Applies an inverse transform to the transformed data to obtain the data in real space.

        Parameters
        ----------
        xfft : xarray DataArray, optional
            The tranformed data to invert. Default is None, in which case the data is self.xfft,
            which results from self.transform()

        Return
        ------
        xinv : xarray DataArray
            The inverted, filtered data
        '''

        if(xfft is None):
            xfft = self.xfft
        
        # invert the filtered data
        xinv = np.fft.ifft(xfft, axis=self.timedim) * xfft.frequency.size
        xinv = np.fft.ifft(xinv, axis=self.londim) * xfft.wavenumber.size
        xinv = 2 * np.real(xinv)
        xinv = xr.DataArray(xinv, dims=dims, coords=coords)

        if(xfft == self.xfft)
            self.xinv = xinv
        return xinv

    # --------------------------------------------------

    def filter(kmin=None, kmax=None, pmin=None, pmax=None, xfft=None):
        '''
        Filters the transformed data on a box in fourier space, defined by
        lower and upper bounds in frequency and wavenumber. The arguments
        (kmin, kmax) specify the filter bounds in wavenumber, and the arguments
        (pmin, pmax) specify the filter bounds in frequency. All four arguments
        are 'None' by default. If e.g. (kmin, kmax) are specified, but (pmin,pmax)
        are not, then the filtering is done only in wavenumber. At least one
        of these argument pairs must be passed.
        
        Parameters
        ----------
        kmin : int, optional
            Minimum wavenumber included in the output. Default is None.
        kmax : int, optional
            Maximum wavenumber included in the output. Default is None.
        pmin : int, optional
            Minimum period (in days) included in the output. Default is None.
        pmax : int, optional
            Maximum period (in days) included in the output. Default is None.
        xfft : xarray DataArray, optional
            The tranformed data to filter. Default is None, in which case the data is self.xfft,
            which results from self.transform()

        Returns
        -------
        xfilt : xarray DataArray
            The filtered data in fourier space
        '''

        # check arguments
        if(xfft is None):
            xfft = self.xfft
        fktot = [f is None for f in [kmin, kmax]]
        fptot = [f is None for f in [pmin, pmax]]
        assert(fktot == 0 or fktot == 2), 'Must pass both kmin and kmax, or neither'
        assert(fptot == 0 or fptot == 2), 'Must pass both pmin and pmax, or neither'
        assert(pmin>=0 and pmax>=0), 'Requested period range (pmin, pmax) must be positive'
        assert(pmin < pmax), 'pmin must be less than pmax'
        assert(kmin < kmax), 'kmin must be less than kmax'
         
        # convert min/max period to min/max freq
        self.fmin, self.fmax = 1/pmax, /pmin
        
        # ensure fft has been run
        assert(self.xfft is not None): 'transform() must be run before filtering'

        # select desired wavenumbers and frequencies
        xfilt = xfft.where(((xfft.wavenumber >= kmin) & (xfft.wavenumber <= kmax)) & \
                           ((xfft.frequency >= fmin) & (xfft.frequency <= fmax)), 0)
        return xfilt
   
    # -------------------------------------------------- 

    def filter_westerly(self):
        '''
        Filters the transformed data to return only westerly (eastward propegating) waves
        
        Returns
        -------
        xw : xarray DataArray
            The filtered westerly data in fourier space
        '''
       
        xw = self.filter(1, np.max(self.xfft.wavenumber), 
                         0, np.max(self.xfft.frequency))
        return xw 

    def filter_easterly(self):
        '''
        Filters the transformed data to return only easterly (westward propegating) waves
        
        Returns
        -------
        xe : xarray DataArray
            The filtered easterly data in fourier space
        '''
       
        xe = self.filter(np.min(self.xfft.wavenumber), -1, 
                         0, np.max(self.xfft.frequency))
        return xe 

    # --------------------------------------------------

    def invert_filtered(kmin=None, kmax=None, pmin=None, pmax=None):
        '''
        Filters the transformed data on a box in fourier space, defined by
        lower and upper bounds in frequency and wavenumber, and inverts the result
        back to real space. The arguments (kmin, kmax) specify the filter bounds in 
        wavenumber, and the arguments (pmin, pmax) specify the filter bounds in 
        frequency. All four arguments are 'None' by default. If e.g. (kmin, kmax) are 
        specified, but (pmin,pmax) are not, then the filtering is done only in 
        wavenumber. At least one of these argument pairs must be passed.

        Parameters
        ----------
        kmin : int
            Minimum wavenumber included in the output
        kmax : int
            Maximum wavenumber included in the output
        pmin : int
            Minimum period (in days) included in the output
        pmax : int
            Maximum period (in days) included in the output

        Return
        ------
        xinv : xarray DataArray
            The inverted, filtered data
        '''

        # filter the data
        xfilt = self.filter(kmin, kmax, pmin, pmax)
        # invert the filtered data
        xinv = self.invert(xfilt)
        return xinv
    
    # --------------------------------------------------

    def invert_westerly_filtered(self):
        '''
        Filters the transformed data to return only westerly (eastward propegating) waves, 
        and then inverts the result back to real space

        Return
        ------
        xwinv : xarray DataArray
            The inverted, filtered westerly data
        '''

        # filter the data
        xfilt = self.filter_westerly()
        # invert the filtered data
        xinv = self.invert(xfilt)
        return xinv
    
    def invert_easterly_filtered(self):
        '''
        Filters the transformed data to return only easterly (westward propegating) waves, 
        and then inverts the result back to real space

        Return
        ------
        xwinv : xarray DataArray
            The inverted, filtered easterly data
        '''

        # filter the data
        xfilt = self.filter_easterly()
        # invert the filtered data
        xinv = self.invert(xfilt)
        return xinv


# ==============================================================================
# ==============================================================================


# The following functions are users of the wave_spectrum class defined above.
# They provide several convenient capabilities without needing to construct 
# wave_spectrum objects directly

def cospectrum(x1, x2, time_bounds=None, lat_band=None, ftaper=None, return_quadrature=False):
    '''
    Computes the cospectrum x1x2*, where x2* is the complex conjugate of x2.

    Parameters
    ----------
    x1 : xarray DataArray
        The first dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    x2 : xarray DataArray
        The second dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    time_bounds : list of float
        List of length 2, giving the left and right bounds of the time window, in days. 
        The subset of data lying outside of this window will be discarded. Default is 
        None, in which the full time domain of the data is used.
    lat_band : float, optional
        List of length 2, giving the southern and northern bounds of the latitude band, 
        in degrees. The subset of data lying outside of this band will be discarded. 
        Default is None, in which the full latitude domain of the data is used.
    ftaper : float, optional
        The fraction of the data that's used for tapering the data in time.
        ftaper/2 will be tapered at each end. Default is 0.1.
    return_quadrature : boolean, optional
        If True, return both the real part of the cross-spectral density (the cospectrum)
        and the imaginary part of the cross-spectral density (the quadrature)
        If False, return only the cospectrum.
        Default is False.
    
    Returns
    -------
    xarray DataArray
        The cospectrum x1x2*
    OR
    list
        [the cospectrum of x1x2*, the quadrature of x1x2*]
    '''
    
    # construct objects
    ws1 = wave_spectrum(x1, time_bounds, lat_band)
    ws2 = wave_spectrum(x2, time_bounds, lat_band)

    # transform data
    ws1.transform(ftaper)
    ws2.transform(ftaper)

    # get cross-spectral density, cospectrum
    cds = ws1.xfft * ws2.xfft.conjugate()
    cospectrum = cds.real()
    quadrature = cds.imag()
    
    # return
    if(return_quadrature): return cospectrum, quadrature
    else: return cospectrum


# --------------------------------------------------


def windowed_timeseries_spectrum(x, window_length, window_overlap, lat_band=None, ftaper=None)
    '''
    Performs 2-dimensional fourier transforms of the data x for multiple time windows. The time
    windows span the time series of the data, and each overlap eachother by a specified amount.
    The right end (larger time) of window i will overlap the left end (smaller time) of window i+1.
    The total number of windows constructed will be (tf/(window_length-window_overlap)), where tf
    is the final time of the input time series

    Parametrers
    -----------
    window_width : int, optional
        Width of temporal window to use for the spectral analysis, in number of days.
        A power spectrum will be produced for every time period of length 
        (window_width-window_overlap) in the data. Default is 90 days.
    window_overlap : int, optional
        Width of temporal window overlaps, in number of days. Default is 30 days.
    lat_band : list of float
        bounds of latitude band over which to apply the transformation
    ftaper : float, optional
        The fraction of the data that's used for tapering the data in time.
        ftaper/2 will be tapered at each end. Default is 0.1.

    Returns
    -------
    windowed_transform : xarray DataArray
        The transformed data, with dimensions (frequency, wavenumber, window). The coordinate
        attached to the 'window' dimension is a tuple, giving the window bounds, in days
    '''

    # verify arguments
    assert(isinstance(x, xarray.core.dataarray.DataArray)), 'x must be an xarray DataArray'
    assert(window_length > window_overlap), 'window_length must be larger than window_overlap'
    if(lat_band is not None):
        assert(lat_bounds[0] > lat_bounds[1]), 'lat_bounds must be increasing'

    # verify data dimensions
    self.dims = np.array(x.dims)
    assert('time' in self.dims), 'Data must have a \'time\' dimension'
    if(lat_band is not None):
        assert('lat' in self.dims), 'Data must have a \'lat\' dimension'
    
    # build time_bounds
    # todo
    self.spd = self.time.diff('time').values.astype('timedelta64[D]').astype(int)

    # construct object
    for i in range(len(time_windows)):
        ws = wave_spectrum(x, time_windows[i], lat_band)
        transform = ws.transform()
        transform.expand_dims({'window':[time_windows[i]]}) 
        if(i > 0): windowed_transform = xr.merge(windowed_transform, transform)
        else: windowed_transform = transform

    return windowed_transform

# --------------------------------------------------


def filter_product(x1, x2, kmin, kmax, pmin, pmax, ftaper=None):
    '''
    In the case that we want to obtain frequency- or wavenumber-filtered representations
    of a variable product x1*x2, the transform and filltration must be done on each variable
    independently before the multipliation. This function performs that operation for input 
    data arrays x1, x2.
    The arguments (kmin, kmax) specify the filter bounds in wavenumber, and the arguments 
    (pmin, pmax) specify the filter bounds in frequency. All four arguments are 'None' by 
    default. If e.g. (kmin, kmax) are specified, but (pmin,pmax) are not, then the filtering 
    is done only in wavenumber. At least one of these argument pairs must be passed.

    Parameters
    ----------
    x1 : xarray DataArray
        The first dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    x2 : xarray DataArray
        The second dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    kmin : int
        Minimum wavenumber included in the output
    kmax : int
        Maximum wavenumber included in the output
    pmin : int
        Minimum period (in days) included in the output
    pmax : int
        Maximum period (in days) included in the output
    ftaper : float, optional
        The fraction of the data that's used for tapering the data in time.
        ftaper/2 will be tapered at each end. Default is 0.1.
    
    Returns
    -------
    filtered_product : xarray DataArray
        The product x1*x2 filtered on (kmin, kmax) and (pmin, pmax) on wavenumber and 
        frequency, respectively
    '''
    
    # construct objects
    ws1 = wave_spectrum(x1)
    ws2 = wave_spectrum(x2)

    # transform data
    ws1.transform(ftaper)
    ws2.transform(ftaper)

    # filter and invert
    x1inv = ws1.invert_filtered(kmin, kmax, pmin, pmax)
    x2inv = ws2.invert_filtered(kmin, kmax, pmin, pmax)

    # get product, return
    filtered_product = x1inv * x2inv
    return filtered_product


# --------------------------------------------------


def filter_product_westerly(x1, x2, ftaper=None):
    '''
    Perfoms the same operation as filter_product(), with an automatic filter on wavenumber for 
    westerly (eastward propegating) waves. See docstrings of filter_product().

    Parameters
    ----------
    x1 : xarray DataArray
        The first dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    x2 : xarray DataArray
        The second dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    ftaper : float, optional
        The fraction of the data that's used for tapering the data in time.
        ftaper/2 will be tapered at each end. Default is 0.1.

    Returns
    -------
    filtered_product : xarray DataArray
        The product x1*x2 filtered on for westerly wavenumbers
    '''
    # construct objects
    ws1 = wave_spectrum(x1)
    ws2 = wave_spectrum(x2)

    # transform data
    ws1.transform(ftaper)
    ws2.transform(ftaper)

    # filter and invert
    x1inv = ws1.invert_westerly_filtered()
    x2inv = ws2.invert_westerly_filtered()

    # get product, return
    filtered_product = x1inv * x2inv
    return filtered_product


# --------------------------------------------------


def filter_product_easterly(x1, x2, ftaper=None):
    '''
    Perfoms the same operation as filter_product(), with an automatic filter on wavenumber for 
    easterly (westwardward propegating) waves. See docstrings of filter_product().

    Parameters
    ----------
    x1 : xarray DataArray
        The first dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    x2 : xarray DataArray
        The second dataset of the product. Dimensions must include 'time' and 'lon', and 
        any other dimensions
    ftaper : float, optional
        The fraction of the data that's used for tapering the data in time.
        ftaper/2 will be tapered at each end. Default is 0.1.

    Returns
    -------
    filtered_product : xarray DataArray
        The product x1*x2 filtered on for easterly wavenumbers
    '''
    # construct objects
    ws1 = wave_spectrum(x1)
    ws2 = wave_spectrum(x2)

    # transform data
    ws1.transform(ftaper)
    ws2.transform(ftaper)

    # filter and invert
    x1inv = ws1.invert_easterly_filtered()
    x2inv = ws2.invert_easterly_filtered()

    # get product, return
    filtered_product = x1inv * x2inv
    return filtered_product
