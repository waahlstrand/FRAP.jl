using ImageFiltering: imfilter!, imfilter, Kernel, centered
using Flux: SamePad, MeanPool


abstract type AbstractROI{T<:Real, S<:Integer} end

struct CircleROI{T, S} <: AbstractROI{T, S}

    x::S
    y::S
    r::T

end 

struct RectangleROI{T, S} <: AbstractROI{T, S}

    x::S
    y::S
    lx::T
    ly::T

end

# Not implemented
# struct ArbitraryROI <: AbstractROI end

function create_bleach_region(x::S, y::S, r::T) where {T<:Real, S<:Integer}

    return CircleROI(x, y, r)

end

function create_bleach_region(x::T, y::T, lx::T, ly::T) where {T<:Real}

    return RectangleROI(x, y, lx, ly)
    
end

function bleach!(c::AbstractArray{Float64, 2}, masks::Array)
    for mask in masks
        c = c .* mask
    end

    return c
end




function create_imaging_bleach_mask(β::T, n_pixels::S, n_pad_pixels::S) where {T<:Real, S<:Integer}

    mask    = ones((n_pixels + 2*n_pad_pixels, n_pixels + 2*n_pad_pixels))
    mask[n_pad_pixels+1:end-n_pad_pixels, n_pad_pixels+1:end-n_pad_pixels] .= β

    return mask

end

function create_bleach_mask(α::T, γ::T, n_pixels::S, n_pad_pixels::S, bleach_region::AbstractROI{T, S}) where {T<:Real, S<:Integer}

    upsampling_factor = 15; # Needs to be a multiple of 3 due the 'box' method in imresize.

    # Create an upsampled bleach region to later downsample
    bounds              = get_bounds(bleach_region, γ)
    bleach_region_mask  = get_bleach_region_mask(bounds, bleach_region, α, upsampling_factor)

    # Smooth mask
    if γ > 0.0
        bleach_region_mask = filter_mask!(bleach_region_mask, γ, upsampling_factor)
    end

    # Downsample mask by the upsampling factor
    bleach_region_mask = downsample_mask!(bleach_region_mask, upsampling_factor) 

    mask    = ones((n_pixels + 2*n_pad_pixels, n_pixels + 2*n_pad_pixels))
    mask[n_pad_pixels+bounds.lb_x:n_pad_pixels+bounds.ub_x, n_pad_pixels+bounds.lb_y:n_pad_pixels+bounds.ub_y] .= bleach_region_mask

    return mask

end

function filter_mask!(x::AbstractArray{Float64, 2}, γ::Float64, upsampling_factor::Int64)

    # Define a covariance kernel
    σ = upsampling_factor * γ
    ℓ = convert(Int64, 4 * ceil(2 * upsampling_factor * γ) + 1 )

    #kernel = centered(Kernel.gaussian((σ,), (ℓ,)))

    #small_mask = imfilter(small_mask, kernel, "replicate")
end

function downsample_mask!(x::AbstractArray{T,2}, k::S) where {T<:Real, S<:Integer}

    dims = size(x)
    target_dims = size(x) .÷ k

    # Initialize a convolution with an averaging kernel
    # Downside: Precompilation of flux is incredibly slow. Consider finding alternatives.
    pool = MeanPool((k, k); pad=0, stride=(k, k))

    # Expand size of image to 4D
    x = reshape(x, (dims..., 1, 1))

    # Downsample
    x = pool(x)

    # Squeeze dimensions
    x = reshape(x, target_dims)

    return x
    
end

function get_bleach_region_mask(bounds::NamedTuple{(:lb_x, :lb_y, :ub_x, :ub_y),NTuple{4,S}}, 
                                bleach_region::AbstractROI{T, S}, 
                                α::T,
                                upsampling_factor::S) where {T<:Real, S<:Integer}

    x = range(bounds.lb_x-1, stop=bounds.ub_x, length=upsampling_factor*(bounds.ub_x-bounds.lb_x+1))
    y = range(bounds.lb_y-1, stop=bounds.ub_y, length=upsampling_factor*(bounds.ub_y-bounds.lb_y+1))

    X, Y = meshgrid(x, y)

    inds = get_bleach_region_index(bleach_region, X, Y)

    mask = ones(size(X))
    mask[inds] .= α

    return mask

end

function get_bounds(bleach_region::RectangleROI{T, S}, γ::T) where {T<:Real, S<:Integer}

    factor = 8

    lb_x = floor(0.5 + bleach_region.x - 0.5 * bleach_region.lx - factor*γ)
    ub_x = ceil(0.5 + bleach_region.x + 0.5 * bleach_region.lx + factor*γ)
    lb_y = floor(0.5 + bleach_region.y - 0.5 * bleach_region.ly - factor*γ)
    ub_y = ceil(0.5 + bleach_region.y + 0.5 * bleach_region.ly + factor*γ)


    lb_x, lb_y, ub_x, ub_y = rescale_bounds(lb_x, lb_y, ub_x, ub_y)

    return (lb_x=lb_x, lb_y=lb_y, ub_x=ub_x, ub_y=ub_y)

end



function get_bounds(bleach_region::CircleROI{T, S}, γ::T) where {T<:Real, S<:Integer}

    factor = 8

    lb_x = floor(0.5 + bleach_region.x - bleach_region.r - factor*γ)
    ub_x = ceil(0.5 + bleach_region.x + bleach_region.r + factor*γ)
    lb_y = floor(0.5 + bleach_region.y - bleach_region.r - factor*γ)
    ub_y = ceil(0.5 + bleach_region.y + bleach_region.r + factor*γ)

    lb_x, lb_y, ub_x, ub_y = rescale_bounds(lb_x, lb_y, ub_x, ub_y)

    return (lb_x=lb_x, lb_y=lb_y, ub_x=ub_x, ub_y=ub_y)

end


function rescale_bounds(lb_x::T, lb_y::T, ub_x::T, ub_y::T) where {T<:Real}
    
    # Bounds must be cast to integers
    lb_x, lb_y, ub_x, ub_y = map(x -> convert(Int64, x), (lb_x, lb_y, ub_x, ub_y))

    lb_x -= 1
    lb_y -= 1
    ub_x += 1
    ub_y += 1

    return (lb_x=lb_x, lb_y=lb_y, ub_x=ub_x, ub_y=ub_y)

end


function get_bleach_region_index(bleach_region::RectangleROI{T, S}, X::AbstractArray{T, 2}, Y::AbstractArray{S, 2}) where {T<:Real, S<:Integer}

    # Make a rectangular cutout of a matrix
    idx_bleach = ((X .>= bleach_region.x .- 0.5 .* bleach_region.lx) .&
                  (X .<= bleach_region.x .+ 0.5 .* bleach_region.lx) .&
                  (Y .>= bleach_region.y .- 0.5 .* bleach_region.ly) .&
                  (Y .<= bleach_region.y .+ 0.5 .* bleach_region.ly))
    
    return idx_bleach

end


function get_bleach_region_index(bleach_region::CircleROI{T, S}, X::AbstractArray{T, 2}, Y::AbstractArray{T, 2}) where {T<:Real, S<:Integer}

    # Make a circular cutout of a matrix
    idx_bleach = (( X .- bleach_region.x ).^2 + ( Y .- bleach_region.y ).^2 .<= bleach_region.r^2)

    return idx_bleach

end

function create_fourier_grid(n_pixels::T) where {T<:Integer}

    # Julia has no ndgrid or meshgrid function, out of principle
    # it would seem. Not "Julian" enough.
    x = range(-n_pixels ÷ 2, length=n_pixels)
    y = range(-n_pixels ÷ 2, length=n_pixels)

    # List comprehensions are very quick
    X = [j for i in x, j in y]
    Y = [i for i in x, j in y]

    X *= (2π / n_pixels)
    Y *= (2π / n_pixels)

    # Centre the transform
    ξ² = ifftshift(X.^2 + Y.^2)

    return ξ²
end



function meshgrid(x, y)

    # List comprehensions are very quick
    X = [i for i in x, j in y]
    Y = [j for i in x, j in y]

    return X, Y
end # function
