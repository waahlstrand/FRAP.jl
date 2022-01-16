using LinearAlgebra
using CUDA, CUDA.CUFFT
using FFTW
using Random


function signal(experiment::ExperimentParams{T}, bath::BathParams{T}) where {T<:Real}

    # Select device
    if CUDA.functional() && experiment.device == "gpu"
        cpu_or_gpu = function(x::AbstractArray) 
            return CuArray(x)
        end
    elseif (experiment.device == "cpu")
        cpu_or_gpu = function(x::AbstractArray) 
            return x
        end    else 
        @warn "Invalid device; falling back on cpu."
        cpu_or_gpu = function(x::AbstractArray) 
            return x
        end    
    end


    # Rescale the diffusion coefficient from SI to per pixel²
    Dₚ  = experiment.D / bath.pixel_size^2 

    # Create masks
    imaging    = imaging_mask(experiment.β, bath.n_pixels, bath.n_pad_pixels) |> cpu_or_gpu
    bleaching  = bleaching_mask(experiment.α, experiment.γ, bath.n_pixels, bath.n_pad_pixels, bath.ROI_pad) |> cpu_or_gpu

    # Calculate the dimensions needed for
    # the real FFT
    ds = (div(size(bath.ξ², 1),2)+1,size(bath.ξ², 2))

    # Initialize a concentration
    mobile, immobile = concentration(experiment.c₀, experiment.ϕₘ, (bath.n_elements, bath.n_elements, bath.n_frames))

    mobile    = mobile |> cpu_or_gpu
    immobile  = immobile |> cpu_or_gpu

    # Pre-allocate the FFT output
    ĉ  = zeros(Complex{T}, ds) |> cpu_or_gpu

    # Pre-plan the Fast Fourier Transforms
    P, P̂ = ffts(bath.ξ², ds, cpu_or_gpu)
    A    = kernel(bath.ξ², Dₚ, experiment.δt, ds) |> cpu_or_gpu

    # Define when to bleach with what
    stages = bleaching_stages(bleaching, imaging, bath)

    # Simulate time evolution
    for stage in stages

        evolve!(mobile, immobile, A, stage..., ĉ, P, P̂)

    end

    # Get only concentration when not bleaching
    # and without padding
    c = extract_concentration((mobile, immobile), bath)

    # Add noise
    n = noise(experiment.a, experiment.b, size(c)) |> cpu_or_gpu
    c = sqrt.(c).*n

    return c
    
end

function extract_concentration(cs, bath::BathParams{T}) where {T <: Real}
    
    # Sum mobile and immobile parts
    c = sum(cs)

    # Remove bleach
    bleach_idx = (bath.n_prebleach_frames+1):(bath.n_prebleach_frames+bath.n_bleach_frames)

    # Remove padding
    nonpadding_idx = (bath.n_pad_pixels+1):(bath.n_pixels+bath.n_pad_pixels)

    c = c[nonpadding_idx, nonpadding_idx,  setdiff(1:end, bleach_idx)]

    return c


end

function bleaching_stages(bleaching, imaging, bath::BathParams{T}) where {T<: Real}
    
    slices = (1:bath.n_prebleach_frames,
              bath.n_prebleach_frames:bath.n_prebleach_frames+bath.n_bleach_frames,
              bath.n_prebleach_frames+bath.n_bleach_frames:bath.n_frames-1)


    stages = (
                ([imaging], slices[1]), 
                ([imaging, bleaching], slices[2]), 
                ([imaging], slices[3])
              )
  
    return stages

end


function ffts(ξ²::Array{T, 2}, ds, cpu_or_gpu::Function) where {T <: Real}

    plan     = zeros(T, size(ξ²)) |> cpu_or_gpu
    inv_plan = zeros(Complex{T},ds) |> cpu_or_gpu

    P = plan_rfft(plan) 
    P̂ = plan_irfft(inv_plan, size(plan, 1)) 

    return P, P̂

end

function noise(a, b, dims)

    gaussian = randn(dims)

    return sqrt(a+b)*gaussian

end

function evolve!(mobile::AbstractArray{T, 3}, 
    immobile::AbstractArray{T, 3},
    A::AbstractArray{T, 2},
    masks, 
    frames::UnitRange{S}, ĉ, P, P̂) where {T<:Real, S<:Integer}


    # Initial concentration before diffusion
    mobile_frame   = mobile[:, :, first(frames)]
    immobile_frame = immobile[:, :, first(frames)]

    # Diffuse for n_frames timesteps
    for frame in frames

        # Make one step of length δt
        mobile_frame .= diffuse!(A, mobile_frame, ĉ, P, P̂)

        # Apply imaging bleach masks
        mobile_frame    .= bleach!(mobile_frame, masks)
        immobile_frame  .= bleach!(immobile_frame, masks)

        # Save time evolution
        mobile[:, :, frame+1]   = mobile_frame
        immobile[:, :, frame+1] = immobile_frame

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