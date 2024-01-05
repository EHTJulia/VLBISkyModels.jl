# Base API



## Contents

```@contents
Pages = ["base_api.md"]
```

## Index

```@index
Pages = ["base_api.md"]
```

```@meta
CurrentModule = ComradeBase
```

## Model API

```@docs
ComradeBase.flux
ComradeBase.visibility
ComradeBase.visibilities
ComradeBase.visibilities!
ComradeBase.intensitymap
ComradeBase.intensitymap!
ComradeBase.IntensityMap
ComradeBase.amplitude(::Any, ::Any)
ComradeBase.amplitudes
ComradeBase.bispectrum
ComradeBase.bispectra
ComradeBase.closure_phase
ComradeBase.closure_phases
ComradeBase.logclosure_amplitude
ComradeBase.logclosure_amplitudes
PolarizedTypes.mpol(::ComradeBase.AbstractPolarizedModel, ::Any)
PolarizedTypes.polellipse(::ComradeBase.AbstractPolarizedModel, ::Any)
PolarizedTypes.polarization(::ComradeBase.AbstractPolarizedModel, ::Any)
PolarizedTypes.fracpolarization(::ComradeBase.AbstractPolarizedModel, ::Any)
PolarizedTypes.mbreve(::ComradeBase.AbstractPolarizedModel, ::Any)
```

### Model Interface
```@docs
ComradeBase.AbstractModel
ComradeBase.isprimitive
ComradeBase.visanalytic
ComradeBase.imanalytic
ComradeBase.ispolarized
ComradeBase.radialextent
ComradeBase.PrimitiveTrait
ComradeBase.IsPrimitive
ComradeBase.NotPrimitive
ComradeBase.DensityAnalytic
ComradeBase.IsAnalytic
ComradeBase.NotAnalytic
ComradeBase.visibility_point
ComradeBase.visibilities_analytic
ComradeBase.visibilities_analytic!
ComradeBase.visibilities_numeric
ComradeBase.visibilities_numeric!
ComradeBase.intensity_point
ComradeBase.intensitymap_analytic
ComradeBase.intensitymap_analytic!
ComradeBase.intensitymap_numeric
ComradeBase.intensitymap_numeric!
```

### Image Types
```@docs
ComradeBase.IntensityMap(::AbstractArray, ::AbstractGrid)
ComradeBase.StokesIntensityMap
ComradeBase.imagepixels
ComradeBase.RectiGrid
ComradeBase.dims
ComradeBase.named_dims
ComradeBase.axisdims
ComradeBase.stokes
ComradeBase.imagegrid
ComradeBase.fieldofview
ComradeBase.pixelsizes
ComradeBase.phasecenter
ComradeBase.centroid
ComradeBase.second_moment
ComradeBase.header
ComradeBase.NoHeader
ComradeBase.MinimalHeader
ComradeBase.load
ComradeBase.save
```


## Polarization

```@docs
ComradeBase.AbstractPolarizedModel
PolarizedTypes.StokesParams
PolarizedTypes.ElectricFieldBasis
PolarizedTypes.RPol
PolarizedTypes.LPol
PolarizedTypes.XPol
PolarizedTypes.YPol
PolarizedTypes.PolBasis
PolarizedTypes.CirBasis
PolarizedTypes.LinBasis
PolarizedTypes.CoherencyMatrix
PolarizedTypes.evpa
PolarizedTypes.m̆
PolarizedTypes.linearpol
PolarizedTypes.innerprod
PolarizedTypes.basis_components
PolarizedTypes.basis_transform
PolarizedTypes.polarization
PolarizedTypes.fracpolarization
```

## Internal Methods not part of public API
```@docs
ComradeBase._visibilities
ComradeBase._visibilities!
```
