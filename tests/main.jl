
using Random
using FRAP
using Plots
using Dates

function main()

    Random.seed!(1234)

    #############################################
    # Parameters
    # Will be defined in configuration files

    n_pixels     = 256
    n_pad_pixels = 64#128]
    pixel_size::Float32   = 7.5e-7

    n_prebleach_frames    = 10
    n_bleach_frames       = 1
    n_postbleach_frames   = 100

    lx::Float32 = 30e-6 
    ly::Float32 = 30e-6
    x::Float32 = 128.0
    y::Float32 = 128.0
    r::Float32 = 15e-6


    c₀::Float32 = 0.95
    ϕₘ::Float32 = 1.0
    D_SI::Float32 = 1e-11; # m^2/s

    δt::Float32 = 0.1

    α::Float32 = 0.7
    β::Float32 = 1.0
    γ::Float32 = 0.0
    a::Float32 = 0.02
    b::Float32 = 0.0
    device::String = "gpu"

    #############################################
    # Run the experiment

    experiment  = ExperimentParams(; c₀, ϕₘ, D=D_SI, δt, α, β, γ, a, b, device=device)
    bath        = BathParams(x, y, r; n_pixels, n_pad_pixels, pixel_size, 
                                      n_prebleach_frames, n_bleach_frames, n_postbleach_frames)

    @info "Generating signal..."
    c = signal(experiment, bath) |> Array
    @info "Signal generated!"

    @info "Calculating recovery curve..."
    rc = recovery_curve(c, bath) 
    @info "Recovery curve done!"

    @info "Creating animation..."
    animation = @animate for i in 1:size(c, 3)
        h = heatmap(c[:,:,i], clim=(0,1); xlim=(0,n_pixels), ylim=(0,n_pixels), colorbar=nothing)
        p = plot(1:i, rc[1:i], ylim=(0.5,1), xlim=(0,size(c,3)-1), legend=false)
        plot(h, p, layout=grid(1, 2, widths=[0.5, 0.5], heights=[0.75, 0.75]) )
    end
    date = Dates.format(now(), "dd-mm-yyyy_HHMM")
    filename = "$(@__DIR__)/images/$(date).gif"
    gif(animation, filename)

end

main()

# @profview main()

