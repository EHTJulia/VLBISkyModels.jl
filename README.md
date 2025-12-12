# VLBISkyModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ehtjulia.github.io/VLBISkyModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ehtjulia.github.io/VLBISkyModels.jl/dev/)
[![Build Status](https://github.com/EHTJulia/VLBISkyModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EHTJulia/VLBISkyModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/EHTJulia/VLBISkyModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EHTJulia/VLBISkyModels.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

This package contains simple geometric and generic imaging models for on-sky radio emission. It also defined the infrastructure so that people can add their own models and have them work with 
downstream packages like [VIDA.jl](https://github.com/ptiede/VIDA.jl) and [Comrade.jl](https://github.com/ptiede/Comrade.jl). The on-sky models currently have polarized, spatial-temporal-spectral
components. 
