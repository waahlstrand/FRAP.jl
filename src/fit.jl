
# Pixel-wise
function residual(c::AbstractArray{T, 3}, experiment, bath, rng) where {T<:Real}

    # TODO: Must be mutable somehow
    experiment.a = 0
    experiment.b = 0

    # Calculate a model signal
    c̃ = FRAP.run(experiment, bath, rng) 

    # Calculate residuals
    r = residual(c, c̃)

    return r

    
end

# Recovery curve
function residual(rc::AbstractArray{T, 1}, experiment, bath, rng) where {T<:Real}

    # Calculate a model signal
    c̃ = FRAP.run(experiment, bath, rng)
    rc̃ = FRAP.recovery_curve(c̃, bath) 

    # Calculate 
    r = residual(rc, rc̃)

    return r
    
end

function residual(x::T, x̃::T) where {T<:AbstractArray}

    r = x .- x̃ |> vec
    
    return r

end


# function fit(c::Array{T, 2}, experiment, bath, fitting, rng, mode=:pixel_wise) where {T<:Real}
    
#     if mode == :recovery_curve

#         rc = FRAP.recovery_curve(c, bath.bleach_mask)
#         r = residual(rc, experiment, bath, rng)

#     elseif mode == :pixel_wise

#         r = residual(rc, experiment, bath, rng)

#     end

#     for fit in 1:fitting.n_fits


#     end



# end