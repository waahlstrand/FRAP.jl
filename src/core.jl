using LinearAlgebra
# using CUDA, CUDA.CUFFT
using FFTW
using Random

# device(x) = (parse(Int,ENV["CUDA_VISIBLE_DEVICES"]) > 0 && CUDA.functional(true)) ? cu(x) : x

@with_kw struct Signal{T<:Real}
    experiment::ExperimentParams{T}
    bath::BathParams{T}
end

function (s::Signal)()

    return run(s.experiment, s.bath)

end


function run(experiment::ExperimentParams{T}, bath::BathParams{T}) where {T<:Real}


    # Rescale the diffusion coefficient from SI to per pixel²
    Dₚ  = experiment.D / bath.pixel_size^2 

    # Create masks
    imaging_mask    = FRAP.create_imaging_bleach_mask(experiment.β, bath.n_pixels, bath.n_pad_pixels) # |> device
    bleach_mask     = FRAP.create_bleach_mask(experiment.α, experiment.γ, bath.n_pixels, bath.n_pad_pixels, bath.ROI_pad) # |> device

    # Calculate the dimensions needed for
    # the real FFT
    ds = (div(size(bath.ξ², 1),2)+1,size(bath.ξ², 2))

    # Initialize a concentration
    cs = concentration(experiment.c₀, experiment.ϕₘ, (bath.n_elements, bath.n_elements, bath.n_frames)) # |> device

    # Pre-allocate the FFT output
    ĉ  = zeros(Complex{T}, ds) # |> device

    # Pre-plan the Fast Fourier Transforms
    P, P̂ = ffts(bath.ξ², ds)
    A    = kernel(bath.ξ², Dₚ, experiment.δt, ds)

    # Define when to bleach with what
    stages = bleaching_stages(bleach_mask, imaging_mask, bath)

    # Simulate time evolution
    for stage in stages

        evolve!(cs..., A, stage..., ĉ, P, P̂)

    end

    # Get only concentration when not bleaching
    # and without padding
    c = get_true_concentration(cs, bath)

    # Add noise
    c .= add_noise(c, experiment.a, experiment.b)

    return c
    
end

function get_true_concentration(cs, bath::BathParams{T}) where {T <: Real}
    
    # Sum mobile and immobile parts
    c = sum(cs)

    # Remove bleach
    bleach = (bath.n_prebleach_frames+1):(bath.n_prebleach_frames+bath.n_bleach_frames)

    # Remove padding
    nonpadding = (bath.n_pad_pixels+1):(bath.n_pixels+bath.n_pad_pixels)

    c = c[nonpadding, nonpadding,  setdiff(1:end, bleach)]

    return c


end

function bleaching_stages(bleach_mask, imaging_mask, bath::BathParams{T}) where {T<: Real}
    
    slices = (1:bath.n_prebleach_frames,
              bath.n_prebleach_frames:bath.n_prebleach_frames+bath.n_bleach_frames,
              bath.n_prebleach_frames+bath.n_bleach_frames:bath.n_frames-1)


    stages = (
                ([imaging_mask], slices[1]), 
                ([imaging_mask, bleach_mask], slices[2]), 
                ([imaging_mask], slices[3])
              )
  
    return stages

end


function ffts(ξ²::Array{T, 2}, ds) where {T <: Real}

    plan     = zeros(T, size(ξ²)) # |> device
    inv_plan = zeros(Complex{T},ds) # |> device

    P = plan_rfft(plan) 
    P̂ = plan_irfft(inv_plan, size(plan, 1)) 

    return P, P̂

end

function add_noise(c, a, b)

    gaussian = randn(size(c)) # |> device

    return c .+ sqrt.(a.+b.*c).*gaussian

end

function evolve!(mobile::AbstractArray{T, 3}, 
    immobile::AbstractArray{T, 3},
    A::AbstractArray{T, 2},
    masks, 
    frames::UnitRange{S}, ĉ, P, P̂) where {T<:Real, S<:Integer}


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

function kernel(ξ²::AbstractArray{T, 2}, D::T, δt::T, ds) where {T<:Real}
    
    ξ² = ξ²[1:ds[1], 1:ds[2]] 
    
    return step(ξ², D, δt)
    
end

function step(ξ²::AbstractArray{T, 2}, D::T, δt::T) where {T<:Real}
    
    return exp.(-D * ξ² * δt)
    
end