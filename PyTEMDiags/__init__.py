# Oops, why are you looking here? PyTEMDiags is implemented as a Python "regular package",
# where each subdirectory in the top-level parent package must be given __init__.py files,
# containing code run at import-time.
# See section 5.2.1 at
# https://docs.python.org/3/reference/import.html#regular-packages
# and the guidelines for minimal Python package structure at
# https://python-packaging.readthedocs.io/en/latest/minimal.html


# =========================================================================


from .sph_zonal_mean import sph_zonal_mean
from .tem_diagnostics import TEMDiagnostics

# define attributes
__version__ = '0.1'
__author__ = 'Joe Hollowed <hollowed@umich.gov>'
