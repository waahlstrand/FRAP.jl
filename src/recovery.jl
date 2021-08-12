using Flux
using Statistics

function recovery_curve(c::AbstractArray{T, 3}, bath::BathParams{T}) where {T<:Real}

    # Create a binary ROI shaped matrix
    ROI = FRAP.create_mask(bath.n_pixels, bath.n_pad_pixels, bath.ROI; type=T) |> gpu
    n_pixels_in_ROI = sum(ROI)
    
    # Summation over each frame matrix in time after multiplying with a ROI shaped matrix
    rc = map(x -> dot(x, ROI), eachslice(c; dims=3))/n_pixels_in_ROI |> gpu
     

    return rc


end