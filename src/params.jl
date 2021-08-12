using Parameters
using YAML

@with_kw struct ExperimentParams{T<:Real}

    c₀::Union{T, Array{T,1}}
    ϕₘ::T
    D::Union{T, Array{T,1}}
    δt::T

    α::Union{T, Array{T,1}}
    β::T
    γ::T

    a::Union{T, Array{T,1}}
    b::T

end

function ExperimentParams(pixel_size::T; 
                          c₀::Union{T, Array{T,1}}, 
                          ϕₘ::T, 
                          D::Union{T, Array{T,1}}, 
                          δt::T, 
                          α::Union{T, Array{T,1}}, 
                          β::T, 
                          γ::T, 
                          a::Union{T, Array{T,1}}, 
                          b::T) where T <: Real

    D = D ./ pixel_size^2

    return ExperimentParams{T}(c₀=c₀, ϕₘ=ϕₘ, D=D, δt=δt, α=α, β=β, γ=γ, a=a, b=b)
end

struct BathParams{T<:Real, S<:Integer, R<:FRAP.AbstractROI{<:Real}}

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

function BathParams(x::T, y::T, r::T; 
    n_pixels::S,  
    n_pad_pixels::S, 
    pixel_size::T,  
    n_prebleach_frames::S,   
    n_bleach_frames::S,     
    n_postbleach_frames::S) where {T<:Real, S<:Integer}

    n_frames   = n_prebleach_frames + n_postbleach_frames + n_bleach_frames
    n_elements = n_pixels + 2*n_pad_pixels


    r /= pixel_size

    ROI = FRAP.create_bleach_region(x, y, r)
    ξ² = convert(Array{T, 2}, FRAP.create_fourier_grid(n_elements))

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

function BathParams(x::T, y::T, lx::T, ly::T;
    n_pixels::S,  
    n_pad_pixels::S, 
    pixel_size::T,  
    n_prebleach_frames::S,   
    n_bleach_frames::S,     
    n_postbleach_frames::S) where {T<:Real, S<:Integer}

    n_frames   = n_prebleach_frames + n_postbleach_frames + n_bleach_frames
    n_elements = n_pixels + 2*n_pad_pixels

    lx /= pixel_size
    ly /= pixel_size

    # ROI = FRAP.create_bleach_region(x+n_pad_pixels, y+n_pad_pixels, lx, ly)
    # ξ² = FRAP.create_fourier_grid(n_elements)

    ROI = FRAP.create_bleach_region(x, y, lx, ly)
    ξ² = convert(Array{T, 2}, FRAP.create_fourier_grid(n_elements))

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



function from_config(file :: String; type = Float32)

    args = YAML.load_file(file)

    exp_args = args["experiment"]
    bath_args = args["bath"]

    for (k, v) in exp_args
        if typeof(v) <: AbstractArray
            exp_args[k] = convert(Array{type,1}, v)
        else
            exp_args[k] = convert(type, v)
        end
    end
    bath_args["x center"] = convert(type, bath_args["x center"] )
    bath_args["y center"] = convert(type, bath_args["y center"] )

    bath_args["radius"] = convert(type, bath_args["radius"] )
    bath_args["pixel size"] = convert(type, bath_args["pixel size"])


    experiment = FRAP.ExperimentParams(bath_args["pixel size"];
                                       c₀ = exp_args["initial concentration"],
                                       ϕₘ = exp_args["mobile fraction"],
                                       D = exp_args["diffusion coefficient"],
                                       δt = exp_args["timestep"],
                                       α = exp_args["bleach"],
                                       β = exp_args["imaging bleach"],
                                       γ = exp_args["gamma"],
                                       a = exp_args["constant variance"],
                                       b = exp_args["linear variance"])

    bath = FRAP.BathParams(bath_args["x center"],
                           bath_args["y center"],
                           bath_args["radius"]; 
                           n_pixels = bath_args["number of pixels"],
                           n_pad_pixels = bath_args["number of pad pixels"],
                           pixel_size = bath_args["pixel size"],
                           n_prebleach_frames = bath_args["number of prebleach frames"],
                           n_bleach_frames = bath_args["number of bleach frames"],
                           n_postbleach_frames = bath_args["number of postbleach frames"],
                           )

    return (experiment, bath)
    
end

@with_kw struct Params{T<:ExperimentParams, S<:BathParams}

    experiment::T
    bath::S

end