# using FFTW, LinearAlgebra
using LinearAlgebra
using Flux, CUDA, CUDA.CUFFT, LinearAlgebra

function run(experiment::ExperimentParams{T}, bath::BathParams{T}, rng) where {T<:Real}


    # Unload all parameters
    c₀ = experiment.c₀
    ϕₘ = experiment.ϕₘ
    D  = experiment.D
    δt = experiment.δt
    α  = experiment.α
    β  = experiment.β
    γ  = experiment.γ
    a  = experiment.a
    b  = experiment.b

    n_prebleach_frames = bath.n_prebleach_frames
    n_bleach_frames = bath.n_bleach_frames
    n_postbleach_frames = bath.n_postbleach_frames
    n_frames = bath.n_frames
    ξ² = bath.ξ²

    dims = (bath.n_elements, bath.n_elements, bath.n_frames)

    # Create masks
    imaging_mask    = FRAP.create_imaging_bleach_mask(β, bath.n_pixels, bath.n_pad_pixels) |> gpu
    bleach_mask     = FRAP.create_bleach_mask(α, γ, bath.n_pixels, bath.n_pad_pixels, bath.ROI) |> gpu

    slices = (1:n_prebleach_frames,
              n_prebleach_frames:n_prebleach_frames+n_bleach_frames,
              n_prebleach_frames+n_bleach_frames:n_frames-1)
    
    # Initialize a concentration
    cs = concentration(c₀, ϕₘ, dims) |> gpu

    # Calculate the dimensions needed for
    # the real FFT
    ds = (div(size(ξ², 1),2)+1,size(ξ², 2))

    # Pre-allocate the FFT output
    ĉ  = zeros(Complex{T}, ds) |> gpu

    # Plan ffts for performance
    # Using real FFTs approximately halfs 
    # memory and time
    plan     = zeros(size(ξ²)) |> gpu
    inv_plan = zeros(Complex{T},ds) |> gpu
    P = plan_rfft(plan) 
    P̂ = plan_irfft(inv_plan, size(plan, 1)) 

    # Calculate the FFT kernel step
    ξ² = ξ²[1:ds[1], 1:ds[2]] 
    A = step(ξ², D, δt) |> gpu

    # Sets the configuration of bleaching and number of frames for bleaching
    stages = (
              ([imaging_mask], slices[1]), 
              ([imaging_mask, bleach_mask], slices[2]), 
              ([imaging_mask], slices[3])
            )

    for stage in stages

        evolve!(cs..., A, stage..., ĉ, P, P̂)

    end

    # Sum mobile and immobile parts
    c = sum(cs)

    # Remove bleach
    bleach = (n_prebleach_frames+1):(n_prebleach_frames+n_bleach_frames)
    c = c[:, :,  setdiff(1:end, bleach)]

    # Add noise
    c .= add_noise(c, a, b, rng)

    return c
    
end

function add_noise(c, a, b, rng)

    gaussian = randn(rng, size(c)) |> gpu

    return c .+ sqrt.(a.+b.*c).*gaussian

end

function evolve!(mobile::AbstractArray{T, 3}, 
    immobile::AbstractArray{T, 3},
    A::AbstractArray{T, 2},
    masks::Array{<:AbstractArray{T, 2}, 1}, frames::UnitRange{S}, ĉ, P, P̂) where {T<:Real, S<:Integer}


    # Initial concentration before diffusion
    c_mobile   = mobile[:, :, first(frames)]
    c_immobile = immobile[:, :, first(frames)]

    # Diffuse for n_frames timesteps
    for frame in frames

        # Make one step of length δt
        c_mobile .= diffuse!(A, c_mobile, ĉ, P, P̂)

        # Apply imaging bleach masks
        c_mobile    .= bleach!(c_mobile, masks)
        c_immobile  .= bleach!(c_immobile, masks)

        # Save time evolution
        mobile[:, :, frame+1]   = c_mobile
        immobile[:, :, frame+1] = c_immobile

    end

    return (mobile, immobile)

end


function diffuse!(A::AbstractArray{T, 2}, c, ĉ, P, P̂) where {T<:Real}

    mul!(ĉ, P, c)   # FFT
    ĉ .= A .* ĉ     # Time step in Fourier domain
    mul!(c, P̂, ĉ)   # Inverse FFT

    return c 

end

function step(ξ²::AbstractArray{T, 2}, D::T, δt::T,) where {T<:Real}

    return exp.(-D * ξ² * δt)
    
end