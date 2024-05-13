```@meta
CurrentModule = VLBISkyModels
```

# VLBISkyModels

`VLBISkyModels` provides an interface and library for of models that can be used to describe the on-sky emission seen by VLBI interferometers. This used to live in the Bayesian VLBI modeling package [Comrade.jl](https://github.com/ptiede/Comrade.jl) but has been recently separated for modularity within the Julia VLBI community. To see how to use VLBISkyModels within `Comrade` see the their [docs](https://ptiede.github.io/Comrade.jl/stable/).

## Contributing

This repository has tries to follow [ColPrac](https://github.com/SciML/ColPrac). If you would like to contribute please feel free to open a issue or pull-request.

## Requirements

The minimum Julia version we require is 1.7. In the future we may increase this as Julia advances.

```@contents
Pages = [
    "index.md",
    "interface.md",
    "api.md"
    "examples/nonanalytic.md"
]
```
