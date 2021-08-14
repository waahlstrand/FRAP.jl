# Source code from Flux.jl/src/functor.jl
# Avoids Flux as a dependency and reimplements 
# their excellent device utility functions

using CUDA
using Adapt: adapt
using Functors: fmap


_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false

# GPU movement convenience function
gpu(x) = CUDA.functional() ? fmap(CUDA.cu, x; exclude = _isbitsarray) : x

# GPU movement convenience function
cpu(m) = (typeof(m) <: CuArray) ? fmap(x -> adapt(Array, x), m) : m
