include(joinpath("src", "FRAP.jl"))

# Packages 
using Flux
using Revise
using Random
using BenchmarkTools
using Plots

function main()

    rng = MersenneTwister(1234);

    #############################################
    # Parameters
    # Will be defined in configuration files

    n_pixels     = 256
    n_pad_pixels = 0#128
    pixel_size::Float32   = 7.5e-7

    n_prebleach_frames    = 10
    n_bleach_frames       = 1
    n_postbleach_frames   = 100
    n_frames              = n_prebleach_frames + n_postbleach_frames
    n_elements            = n_pixels + 2*n_pad_pixels

    lx::Float32 = 30e-6 
    ly::Float32 = 30e-6
    x::Float32 = 128.0
    y::Float32 = 128.0
    r::Float32 = 15e-6


    c₀::Float32 = 0.95
    ϕₘ::Float32 = 1.0
    D_SI::Float32 = 1e-11; # m^2/s
    D = D_SI / pixel_size^2

    δt::Float32 = 0.1

    α::Float32 = 0.7
    β::Float32 = 1.0
    γ::Float32 = 0.0
    a::Float32 = 0.02
    b::Float32 = 0.0

    #############################################
    # Run the experiment

    experiment  = FRAP.ExperimentParams(pixel_size; c₀=c₀, ϕₘ=ϕₘ, D=D_SI, δt=δt, α=α, β=β, γ=γ, a=a, b=b)
    bath        = FRAP.BathParams(x, y, r; 
                                  n_pixels=n_pixels, 
                                  n_pad_pixels=n_pad_pixels, 
                                  pixel_size=pixel_size, 
                                  n_prebleach_frames=n_prebleach_frames,
                                  n_bleach_frames=n_bleach_frames,
                                  n_postbleach_frames=n_postbleach_frames
                                )


    c = FRAP.run(experiment, bath, rng)
    c = c |> cpu

    #############################################
    # Plot the experiment

    theme(:solarized_light)

    @gif for i in 1:size(c, 3)
        heatmap(c[:,:,i], clim=(0,1), aspect_ratio=:equal)
    end
end


