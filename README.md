# PyTEMDiags

This package provides an interface for performing Transformed Eulerian Mean(TEM) analyses on simulated atmospheric datasets. The specific quantities available for computing are those given in Table A1 of the DynVarMIP experimental design protocol ([Gerber & Manzini 2016](https://gmd.copernicus.org/articles/9/3413/2016/), hereafter GM16). This includes the Eliasen-Palm (EP) flux components, the EP flux divergence, TEM residual velocities, the TEM mass streamfunction, and various contributing quantities. See the [Inputs, Outputs](#io) section below for more detail.

Currently, the code supports unstructured ("native") input data, and is not specifically optimized for structured latitude-longitude grids (though such data could always be "raveled" into a generalized "ncol" format). Zonal means, and thus eddy components, are computed on the native grid via a spherical harmonic-based matrix remap approach.

## Getting Started

### Prerequisites

### Installation

### Testing

<div id="io"></div>

## Inputs, Outputs

