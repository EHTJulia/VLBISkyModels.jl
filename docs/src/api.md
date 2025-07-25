# API

## Contents

```@contents
Pages = ["api.md"]
```

## Index

```@index
Pages = ["api.md"]
```

## Model Definitions

### Model Parameter Unpackaing

```@docs
VLBISkyModels.DomainParams
VLBISkyModels.build_param
VLBISkyModels.getparam
VLBISkyModels.@unpack_params
VLBISkyModels.paramtype
```

### Time Frequency Domain

```@docs
VLBISkyModels.TaylorSpectral
```

### Combinators

```@docs
Base.:+(::VLBISkyModels.AbstractModel, ::VLBISkyModels.AbstractModel)
VLBISkyModels.added
VLBISkyModels.convolved
VLBISkyModels.components
VLBISkyModels.smoothed
VLBISkyModels.CompositeModel
VLBISkyModels.AddModel
VLBISkyModels.ConvolvedModel
```

### Geometric and Image Models

```@docs
VLBISkyModels.GeometricModel
VLBISkyModels.ConcordanceCrescent
VLBISkyModels.Crescent
VLBISkyModels.Disk
VLBISkyModels.SlashedDisk
VLBISkyModels.ExtendedRing
VLBISkyModels.Gaussian
VLBISkyModels.TBlob
VLBISkyModels.MRing
VLBISkyModels.Ring
VLBISkyModels.ParabolicSegment
VLBISkyModels.ZeroModel
VLBISkyModels.MultiComponentModel
VLBISkyModels.PolarizedModel
VLBISkyModels.AbstractImageTemplate
VLBISkyModels.RingTemplate
VLBISkyModels.RadialGaussian
VLBISkyModels.RadialDblPower
VLBISkyModels.RadialTruncExp
VLBISkyModels.RadialJohnsonSU
VLBISkyModels.AzimuthalUniform
VLBISkyModels.AzimuthalCosine
VLBISkyModels.GaussianRing
VLBISkyModels.SlashedGaussianRing
VLBISkyModels.EllipticalGaussianRing
VLBISkyModels.EllipticalSlashedGaussianRing
VLBISkyModels.CosineRing
VLBISkyModels.CosineRingwFloor
VLBISkyModels.CosineRingwGFloor
VLBISkyModels.EllipticalCosineRing
VLBISkyModels.LogSpiral
VLBISkyModels.Constant
VLBISkyModels.GaussDisk
VLBISkyModels.ContinuousImage
```

### Image Pulses

```@docs
VLBISkyModels.Pulse
VLBISkyModels.DeltaPulse
VLBISkyModels.BSplinePulse
VLBISkyModels.RaisedCosinePulse
VLBISkyModels.BicubicPulse
VLBISkyModels.Butterworth
```

### Fourier Duality Models

```@docs
VLBISkyModels.FourierDualDomain
VLBISkyModels.xygrid
VLBISkyModels.uvgrid
VLBISkyModels.uviterator
VLBISkyModels.DFTAlg
VLBISkyModels.FFTAlg
VLBISkyModels.NFFTAlg
VLBISkyModels.FINUFFTAlg
VLBISkyModels.InterpolatedModel
```

### Modifiers

```@docs
VLBISkyModels.modify
VLBISkyModels.basemodel
VLBISkyModels.unmodified
VLBISkyModels.renormed
VLBISkyModels.rotated
VLBISkyModels.posangle
VLBISkyModels.shifted
VLBISkyModels.stretched
VLBISkyModels.ModifiedModel
VLBISkyModels.ModelModifier
VLBISkyModels.Stretch
VLBISkyModels.Shift
VLBISkyModels.Rotate
VLBISkyModels.Renormalize
```

### Model Evaluation

For more docstrings on how to evaluate models see [Base API](@ref).

```@docs
```

### Visualization

```@docs
VLBISkyModels.imageviz
VLBISkyModels.polimage
VLBISkyModels.polimage!
VLBISkyModelsMakieExt.PolImage
```

### Misc.

```@docs
VLBISkyModels.center_image
VLBISkyModels.convolve
VLBISkyModels.convolve!
VLBISkyModels.smooth
VLBISkyModels.PoincareSphere2Map
VLBISkyModels.PolExp2Map
VLBISkyModels.regrid
VLBISkyModels.rad2μas
VLBISkyModels.μas2rad
```

### IO

```@docs
VLBISkyModels.load_clean_components
VLBISkyModels.load_fits
VLBISkyModels.save_fits
```

## Internal (Not Public API)

```@docs
VLBISkyModels.AbstractFourierDualDomain
VLBISkyModels.scale_uv
VLBISkyModels.scale_image
VLBISkyModels.transform_uv
VLBISkyModels.transform_image
VLBISkyModels.unpack
VLBISkyModels.stokes_intensitymap
VLBISkyModels.FFTPlan
VLBISkyModels.NUFTPlan
```
