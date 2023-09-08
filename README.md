# PyTEM

This package provides an interface for performing Transformed Eulerian Mean analyses on simulated atmospheric datasets. The specific quantities computed match those given in Table A1 of the DynVarMIP experimental design protocol ([Gerber & Manzini 2016](https://gmd.copernicus.org/articles/9/3413/2016/), hereafter GM16). See [Inputs, Outputs](#io) below for more detail.

Currently, the code supports unstructured input data, and is not ispecifically optimized for structured latitude-longitude grids (though such data could always be "raveled" into a generalzed "ncol" format). Zonal means, and thus eddy components, are computed on the unstructured grid via a spherical harmonic-based matrix remap approach.

## Getting Started

### Prerequisites

### Installation

### Testing

## Inputs, Outputs(#io)

