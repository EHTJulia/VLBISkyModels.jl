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
VLBISkyModels.MRing
VLBISkyModels.Ring
VLBISkyModels.ParabolicSegment
VLBISkyModels.ContinuousImage
VLBISkyModels.ZeroModel
VLBISkyModels.MultiComponentModel
VLBISkyModels.PolarizedModel
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


### Model Image (non analytic FFT)

```@docs
VLBISkyModels.create_cache
VLBISkyModels.modelimage
VLBISkyModels.uviterator
VLBISkyModels.fouriermap
VLBISkyModels.ModelImage
VLBISkyModels.DFTAlg
VLBISkyModels.FFTAlg
VLBISkyModels.FFTCache
VLBISkyModels.NFFTAlg
VLBISkyModels.NUFTCache
VLBISkyModels.ObservedNUFT
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
```


### Misc.
```@docs
VLBISkyModels.center_image
VLBISkyModels.convolve
VLBISkyModels.convolve!
VLBISkyModels.PoincareSphere2Map
VLBISkyModels.regrid
VLBISkyModels.rad2μas
VLBISkyModels.μas2rad
```

## Internal (Not Public API)
```@docs
VLBISkyModels.AbstractCache
VLBISkyModels.scale_uv
VLBISkyModels.scale_image
VLBISkyModels.transform_uv
VLBISkyModels.transform_image
VLBISkyModels.ThreadedModel
VLBISkyModels.unpack
```