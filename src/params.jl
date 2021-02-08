struct ExperimentParams{T<:Real}

    c₀::T
    ϕₘ::T
    D::T
    δt::T

    α::T
    β::T
    γ::T

    a::T
    b::T

    function ExperimentParams(c::T, ϕₘ::T, D_SI::T, δt::T, α::T, β::T, γ::T, a::T, b::T, pixel_size::T) where T

        D = D_SI / pixel_size^2

        return new{T}(c::T, ϕₘ::T, D::T, δt::T, α::T, β::T, γ::T, a::T, b::T)
    end

end

struct BathParams{T<:Real, S<:Integer, R<:FRAP.AbstractROI{<:Real, <:Integer}}

    n_pixels::S   
    n_pad_pixels::S 
    pixel_size::T   
    n_prebleach_frames::S   
    n_bleach_frames::S     
    n_postbleach_frames::S   
    n_frames::S             
    n_elements::S
    ROI::R
    ξ²::Array{T, 2}

end

function BathParams(n_pixels::S,  
    n_pad_pixels::S, 
    pixel_size::T,  
    n_prebleach_frames::S,   
    n_bleach_frames::S,     
    n_postbleach_frames::S,
    x::S,
    y::S,
    r::T) where {T<:Real, S<:Integer}

    n_frames   = n_prebleach_frames + n_postbleach_frames + n_bleach_frames
    n_elements = n_pixels + 2*n_pad_pixels


    r = r/pixel_size

    ROI = FRAP.create_bleach_region(x, y, r)
    ξ² = FRAP.create_fourier_grid(n_elements)

    return BathParams(n_pixels, 
            n_pad_pixels, 
            pixel_size, 
            n_prebleach_frames,
            n_bleach_frames,
            n_postbleach_frames,
            n_frames,
            n_elements, 
            ROI, 
            ξ²) 

end

function BathParams(n_pixels::S,  
    n_pad_pixels::S, 
    pixel_size::T,  
    n_prebleach_frames::S,   
    n_bleach_frames::S,     
    n_postbleach_frames::S,
    x::S,
    y::S,
    lx::T,
    ly::T) where {T<:Real, S<:Integer, R<:FRAP.AbstractROI{<:Real, <:Integer}}

n_frames   = n_prebleach_frames + n_postbleach_frames + n_bleach_frames
n_elements = n_pixels + 2*n_pad_pixels


ROI = FRAP.create_bleach_region(x, y, lx, ly)
ξ² = FRAP.create_fourier_grid(n_elements)

return BathParams(n_pixels::S, 
        n_pad_pixels::S, 
        pixel_size::T, 
        n_prebleach_frames::S,
        n_bleach_frames::S,
        n_postbleach_frames::S,
        n_frames::S,
        n_elements::S,
        ROI::R, 
        ξ²::Array{T, 2})

end

struct Params{T<:ExperimentParams, S<:BathParams}

    experiment::T
    bath::S

end