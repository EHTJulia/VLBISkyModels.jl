name: Documentation

on:
  push:
    branches:
      - main
    tags: '*'
  pull_request:
  
  
concurrency:
  # Skip intermediate builds: always.
  # Cancel intermediate builds: only if it is a pull request build.
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Install Libs
        run: |
          sudo apt-get update
          sudo apt-get install libhdf5-dev
          sudo apt-get install --reinstall libxcb-xinerama0
      - uses: actions/checkout@v2
      - name: Setup python
        uses: actions/setup-python@v1
        with:
          python-version: ${{ matrix.python-version }}
          architecture: ${{ matrix.arch }}
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'
      - name: Install docs dependencies
        run: julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(url="https://github.com/ptiede/ComradeBase.jl"), PackageSpec(path=joinpath(pwd(), "lib/ComradeAHMC")), PackageSpec(path=joinpath(pwd(), "lib/ComradeOptimization")), PackageSpec(path=joinpath(pwd(), "lib/ComradeDynesty")), PackageSpec(path=joinpath(pwd(), "lib/ComradeNested")), PackageSpec(path=joinpath(pwd(), "lib/ComradeAdaptMCMC"))]); Pkg.instantiate()'
      - name: Install examples dependencies
        run: julia --project=examples -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(url="https://github.com/ptiede/ComradeBase.jl"), PackageSpec(path=joinpath(pwd(), "lib/ComradeAHMC")), PackageSpec(path=joinpath(pwd(), "lib/ComradeOptimization")), PackageSpec(path=joinpath(pwd(), "lib/ComradeDynesty")), PackageSpec(path=joinpath(pwd(), "lib/ComradeNested")), PackageSpec(path=joinpath(pwd(), "lib/ComradeAdaptMCMC"))]); Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }} # For authentication with GitHub Actions token
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }} # For authentication with SSH deploy key
        run: julia --project=docs/ docs/make.jl
