# Packages 
# using Revise
using Random
using FRAP
# using BenchmarkTools
using Plots
# theme(:solarized_light)

# includet(joinpath("src", "FRAP.jl"))
# import .FRAP


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

    experiment  = FRAP.ExperimentParams(pixel_size; c₀, ϕₘ, D=D_SI, δt, α, β, γ, a, b)
    bath        = FRAP.BathParams(x, y, r; n_pixels, n_pad_pixels, pixel_size, 
                                n_prebleach_frames, n_bleach_frames, n_postbleach_frames, 
                                )

    # Test the bleach mask creation
    # @benchmark FRAP.create_bleach_mask(α, γ, bath.n_pixels, bath.n_pad_pixels, bath.ROI)

    # Run FRAP
    c = FRAP.run(experiment, bath, rng)

    # Calculate recovery curve
    rc = FRAP.recovery_curve(c, bath)

    # Calculate the residuals
    rc = rc |> cpu
    c  = c |> cpu
    # residual  = FRAP.residual(rc, experiment, bath, rng)
    # plot(1:n_frames, rc)
    # histogram(residual)
    

    # @benchmark FRAP.run($experiment, $bath, $rng)
    #############################################
    # Plot the experiment
    # c = c |> cpu
    @gif for i in 1:size(c, 3)
        heatmap(c[:,:,i], clim=(0,1), aspect_ratio=:equal)
    end

end

main()

# @profview main()

