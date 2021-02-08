include(joinpath("src", "FRAP.jl"))

# Packages 
using Revise
import .FRAP
using Random

rng = MersenneTwister(1234);

#############################################
# Parameters
# Will be defined in configuration files

n_pixels     = 256
n_pad_pixels = 128
pixel_size   = 7.5e-7

n_prebleach_frames    = 10
n_bleach_frames       = 1
n_postbleach_frames   = 100
n_frames              = n_prebleach_frames + n_postbleach_frames
n_elements            = n_pixels + 2*n_pad_pixels

x = 128
y = 128
r = 15e-6


c₀ = 1.0
ϕₘ = 1.0
D_SI = 5e-11; # m^2/s
D = D_SI / pixel_size^2

δt = 0.1

α = 0.6
β = 1.0
γ = 0.0
a = 0.0
b = 0.0

#############################################
# Run the experiment

experiment  = FRAP.ExperimentParams(c₀, ϕₘ, D_SI, δt, α, β, γ, a, b, pixel_size)
bath        = FRAP.BathParams(n_pixels, n_pad_pixels, pixel_size, 
                              n_prebleach_frames, n_bleach_frames, n_postbleach_frames, 
                              x, y, r)


c = FRAP.run(experiment, bath, rng)

#############################################
# Plot the experiment

using Plots
theme(:solarized_light)

@gif for i in 1:size(c, 3)
    heatmap(c[:,:,i], clim=(0,1), aspect_ratio=:equal)
end
