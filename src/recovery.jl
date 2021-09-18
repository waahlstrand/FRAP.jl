using Statistics

# function recovery_curve(c::AbstractArray{T, 3}, bath::BathParams{T}) where {T<:Real}


#     # Create a binary ROI shaped matrix
#     ROI = FRAP.create_mask(bath.n_pixels, bath.n_pad_pixels, bath.ROI; type=T) |> gpu
#     n_pixels_in_ROI = sum(ROI)
    
#     # Summation over each frame matrix in time after multiplying with a ROI shaped matrix
#     rc = map(x -> dot(x, ROI), eachslice(c; dims=3))/n_pixels_in_ROI |> gpu
     

#     return rc

# end


function recovery_curve(c::AbstractArray{T, 3}, bath::BathParams{T}) where {T<:Real}

    rc = zeros(T, (size(c, 3)))
    n_inside = 0
    pixels = 0.5:1:(bath.n_pixels - 0.5)
    
    for k in 1:size(c, 3)

        for (j, y) in enumerate(pixels)

            for (i, x) in enumerate(pixels)

                if FRAP.inside(x, y, bath.ROI)

                    @inbounds rc[k] += c[i, j, k]
                    
                    if k == 1
                        n_inside += 1
                    end

                end

            end
            
        end

    end

    return rc ./ n_inside

end

function recovery_curve(cs::AbstractArray{T, 4}, bath::BathParams{T}) where {T<:Real}

    rcs = zeros(T, (size(c, 3), size(c, 4)))

    for b in 1:size(c, 4)

        @inbounds rcs[:,b] = recovery_curve(cs[:,:,:,b], bath)

    end

    return rcs
    
end