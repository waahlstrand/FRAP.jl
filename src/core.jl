using FFTW: fft, ifft, ifftshift


function run(experiment::ExperimentParams, bath::BathParams, rng)


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
    imaging_mask    = FRAP.create_imaging_bleach_mask(β, bath.n_pixels, bath.n_pad_pixels)
    bleach_mask     = FRAP.create_bleach_mask(α, γ, bath.n_pixels, bath.n_pad_pixels, bath.ROI)

    slices = (1:n_prebleach_frames,
              n_prebleach_frames:n_prebleach_frames+n_bleach_frames,
              n_prebleach_frames+n_bleach_frames:n_frames-1)

    # Initialize a concentration
    cs = concentration(c₀, ϕₘ, dims)

    # Sets the configuration of bleaching and number of frames for bleaching
    stages = (
              ([imaging_mask], slices[1]), 
              ([imaging_mask, bleach_mask], slices[2]), 
              ([imaging_mask], slices[3])
            )

    for stage in stages

        cs = evolve!(cs..., ξ², D, δt, stage...)

    end

    # Sum mobile and immobile parts
    c = sum(cs)

    # Remove bleach
    bleach = (n_prebleach_frames+1):(n_prebleach_frames+n_bleach_frames)
    c = c[:, :,  1:end .!= bleach]

    # Add noise
    c = add_noise(c, a, b, rng)

    return c
    
end

# Evolves an array slice from start to end
function evolve!(mobile::AbstractArray{T, 3}, 
                immobile::AbstractArray{T, 3},
                ξ²::AbstractArray{T, 2},
                D::T,
                δt::T,
                masks::AbstractArray{AbstractArray{T, 2}, 1}, frames::UnitRange{S}) where {T<:Real, S<:Integer}

    
    # Initial concentration before diffusion
    c_mobile   = mobile[:, :, first(frames)]
    c_immobile = immobile[:, :, first(frames)]

    # n_frames   = size(c_mobile, 3)

    # Diffuse for n_frames timesteps
    for frame in frames

        # Make one step of length δt
        c_mobile .= diffuse!(c_mobile, ξ², D, δt)

        # Apply imaging bleach masks
        c_mobile    .= bleach!(c_mobile, masks)
        c_immobile  .= bleach!(c_immobile, masks)

        # Save time evolution
        mobile[:, :, frame+1]   = c_mobile
        immobile[:, :, frame+1] = c_immobile
 
    end

    return (mobile, immobile)

    
end

function add_noise(c, a, b, rng)

    return c .+ sqrt.(a.+b.*c).*randn(rng, size(c))

end

# Performs ordinary linear diffusion
# Dispatch this to add binding or other effects
function diffuse!(c::AbstractArray{T, 2}, ξ²::AbstractArray{T, 2}, D::T, δt::T) where {T<:Real}

    dims = [1, 2]

    ĉ = fft(c, dims)
    ĉ = exp.(-D .* ξ² .* δt) .* ĉ
    c = abs.(ifft(ĉ, dims))

    return c

end