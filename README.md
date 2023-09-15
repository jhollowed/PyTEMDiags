# PyTEMDiags

This package provides an interface for performing Transformed Eulerian Mean(TEM) analyses on simulated atmospheric datasets. The specific quantities available for computing are those given in Table A1 of the DynVarMIP experimental design protocol ([Gerber & Manzini 2016](https://gmd.copernicus.org/articles/9/3413/2016/), hereafter GM16). This includes the Eliasen-Palm (EP) flux components, the EP flux divergence, TEM residual velocities, the TEM mass streamfunction, and various contributing quantities. See the [Inputs, Outputs](#io) section below for more detail.

Currently, the code supports unstructured ("native") input data, and is not specifically optimized for structured latitude-longitude grids (though such data could always be "raveled" into a generalized "ncol" format). Zonal means, and thus eddy components, are computed on the native grid via a spherical harmonic-based matrix remap approach.

## Getting Started

### Installation

1. First, ensure all needed dependencies are installed. If not, they can be installed with a `conda` or `pip`, e.g.
   ```
   pip install numpy>=1.21.5
   pip install xarray>=2022.11.50
   pip install scipy>=1.10.1
   ```
2. Clone the package
   ```
   git clone git@github.com:jhollowed/PyTEMDiags.git
   ```
3. Locally install the package
   ```
   cd PyTEMDiags
   pip install -e .
   ```
   The `-e` flag to `pip install` installs the package with a symlink to the soruce code, so that later updates via `git pull` do not require re-installation.

### Usage
Usage is centered around a `PyTEMDiags.TEMDiagnostics` object, constructed with inputs from a dataset. Once constructed, this object provides an interface for computing all TEM diagnostic quantities of interest via various methods of the class. The following is a minimal example:
```
import PyTEMDiags
import xarray as xr

# --- read wind components U, V, temperature T, vertical pressure velocity OMEGA,
#     pressure P, and latitudes LAT from the data
data = xr.open_dataset('/path/to/netcdf/file.nc')
ua, va, ta, wap, p, lat = data['U'], data['V'], data['T'], data['OMEGA'], data['P'], data['LAT']

# --- construct a diagnostics object
tem = PyTEMDiags.TEMDiagnostics(ua, va, ta, wap, p, lat)
# ---- compute various TEM diagnostics
vtem = tem.vtem()           # the TEM northward residual velocity
wtem = tem.wtem()           # the TEM vertical residual velocity
psitem = tem.psitem()       # the TEM mass stream function
utendwtem = tem.utendwtem() # the tendency of eastward wind due to TEM upward wind advection
```
The methods of `tem` called above are not an exhaustive list of those available; see docstrings in the source code for full lists and descriptions of the class Methods, Attributes, and input parameters.
Note that there are a few assumptions made of the input data variables:
1. Each variable contains at least two, and at most three dimensons, either `(horizontal, vertical)`, or `(horizontal, vertical, time)`. The order of the dimensions do not matter, and the names of the dimensions can be specificed via the argument `dim_names`.
2. Each variable contains coordinates for the vertical and time (if present) dimensions. It does not need to contain a coordinate for the horizontal dimension, as this is assumed to be a unitless integer-valued `ncol`-like quantitity. If the variables *do* contain a horizontal coordinate, this will be ignored, and the input argument `lat` must be passed regardless.
3. The inpute pressure `p` must be provided at the gridpoint level; if the input dataset does not include this variable, it must be computed by the user.

For the assumed units of the input variables and more descriptions, see the docstrings at the top of the `TEMDiagnostics` source code.

### Testing

<div id="io"></div>

## Inputs, Outputs

