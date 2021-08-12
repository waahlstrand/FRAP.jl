using FFTW:ifftshift

abstract type AbstractROI{T <: Real} end

struct CircleROI{T} <: AbstractROI{T}

    x_center::T
    y_center::T
    r::T

end 

struct RectangleROI{T} <: AbstractROI{T}

    x_center::T
    y_center::T
    lx::T
    ly::T

end

# Not implemented
# struct ArbitraryROI <: AbstractROI end

function create_bleach_region(x_center::T, y_center::T, r::T) where {T <: Real}

    return CircleROI(x_center, y_center, r)

end

function create_bleach_region(x_center::T, y_center::T, lx::T, ly::T) where {T <: Real}

    return RectangleROI(x_center, y_center, lx, ly)
    
end

function bleach!(c::AbstractArray{T,2}, masks::Array{<:AbstractArray{T,2}}) where {T <: Real}
    for mask in masks
        c = c .* mask
    end

    return c
end


function create_mask(n_pixels::S, 
                     n_pad_pixels::S,
                     ROI::CircleROI; type=Float32) where {S <: Integer}
    n_elements = n_pixels + 2 * n_pad_pixels

    mask = zeros(type, (n_elements, n_elements))
                 
    x_center = ROI.x_center
    y_center = ROI.y_center
    r        = ROI.r
                 
    pixels = 0.5:1:(n_elements - 0.5)
                 
                 
    r_unit_pixel = 1 / sqrt(2)
                 
    for (j, y) in enumerate(pixels)
        for (i, x) in enumerate(pixels)
                 
            inside_circle   = ( x - x_center )^2 + ( y - y_center )^2 <= (r - r_unit_pixel)^2
            # outside_circle  = ( x - x_center )^2 + ( y - y_center )^2 >= (r + r_unit_pixel)^2
                 
            if inside_circle        # clearly inside circle
                 
                mask[i, j] = 1
                 
            end
        end
    end
                 
    return mask
                 

end


function create_mask(n_pixels::S, 
                     n_pad_pixels::S,
                     ROI::RectangleROI; type=Float32) where {S <: Integer}

    n_elements = n_pixels + 2 * n_pad_pixels

    mask = zeros(type, (n_elements, n_elements))
                 
    x_center = ROI.x_center
    y_center = ROI.y_center
    lx        = ROI.lx
    ly        = ROI.ly

    pixels = 0.5:1:(n_elements - 0.5)

    r_unit_pixel = 1 / sqrt(2)

    for (j, y) in enumerate(pixels)
        for (i, x) in enumerate(pixels)

            inside_rectangle   = (x >=  x_center - 0.5 * lx + 1) & (x <=  x_center + 0.5 * lx - 1) & (y >=  y_center - 0.5 * ly + 1) & (y <=  y_center + 0.5 * ly - 1)
            # outside_rectangle  = (x >=  x_center - 0.5 * lx - 1) & (x <=  x_center + 0.5 * lx + 1) & (y >=  y_center - 0.5 * ly - 1) & (y <=  y_center + 0.5 * ly + 1)

            if inside_rectangle        # clearly inside
                 
                mask[i, j] = 1
                 
            end
        end
    end
                 
    return mask
                 

end





function create_bleach_mask(α::T, 
                            γ::T, 
                            n_pixels::S, 
                            n_pad_pixels::S, 
                            ROI::CircleROI{T}; 
                            subsampling_factor=15) where {T <: Real,S <: Integer}
    
    n_elements = n_pixels + 2 * n_pad_pixels

    mask = ones(T, (n_elements, n_elements))

    x_center = ROI.x_center
    y_center = ROI.y_center
    r        = ROI.r

    pixels = 0.5:1:(n_elements - 0.5)

    sub_pixels = range((1 / (2 * subsampling_factor)) - 0.5, 
                        step=1 / subsampling_factor, 
                        length=(subsampling_factor))

    r_unit_pixel = 1 / sqrt(2)

    for (j, y) in enumerate(pixels)
        for (i, x) in enumerate(pixels)

            inside_circle   = ( x - x_center )^2 + ( y - y_center )^2 <= (r - r_unit_pixel)^2
            outside_circle  = ( x - x_center )^2 + ( y - y_center )^2 >= (r + r_unit_pixel)^2

            if inside_circle        # clearly inside circle

                mask[i, j] = α

            elseif !outside_circle  # Near or on the edge of the bleach region.          
 
                fraction_of_pixel_inside = 0.0

                for x_sub in sub_pixels

                    for y_sub in sub_pixels

                        if ( x + x_sub - x_center )^2 + ( y + y_sub - y_center )^2 <= r^2
                            fraction_of_pixel_inside += 1.0
                        end

                    end

                end

                fraction_of_pixel_inside /= subsampling_factor^2
                mask[i, j] = fraction_of_pixel_inside * α + (1 - fraction_of_pixel_inside) * 1.0

            end
        end
    end

    return mask

end

function create_bleach_mask(α::T, 
                            γ::T, 
                            n_pixels::S, 
                            n_pad_pixels::S, 
                            ROI::RectangleROI{T}; 
                            subsampling_factor=15) where {T <: Real,S <: Integer}
    
    n_elements = n_pixels + 2 * n_pad_pixels

    mask = ones(T, (n_elements, n_elements))

    x_center = ROI.x_center
    y_center = ROI.y_center
    lx        = ROI.lx
    ly        = ROI.ly

    pixels = 0.5:1:(n_elements - 0.5)

    sub_pixels = range((1 / (2 * subsampling_factor)) - 0.5, 
                        step=1 / subsampling_factor, 
                        length=(subsampling_factor))

    r_unit_pixel = 1 / sqrt(2)

    for (j, y) in enumerate(pixels)
        for (i, x) in enumerate(pixels)

            inside_rectangle   = (x >=  x_center - 0.5 * lx + 1) & (x <=  x_center + 0.5 * lx - 1) & (y >=  y_center - 0.5 * ly + 1) & (y <=  y_center + 0.5 * ly - 1)
            outside_rectangle  = (x >=  x_center - 0.5 * lx - 1) & (x <=  x_center + 0.5 * lx + 1) & (y >=  y_center - 0.5 * ly - 1) & (y <=  y_center + 0.5 * ly + 1)

            if inside_rectangle        # clearly inside

                mask[i, j] = α

            elseif !outside_rectangle  # Near or on the edge of the bleach region.          
 
                fraction_of_pixel_inside = 0.0

                for x_sub in sub_pixels

                    for y_sub in sub_pixels

                        if (x_sub >=  x_center - 0.5 * lx) & (x_sub <=  x_center + 0.5 * lx ) & (y_sub >=  y_center - 0.5 * ly) & (y_sub <=  y_center + 0.5 * ly)
                            fraction_of_pixel_inside += 1.0
                        end

                    end

                end

                fraction_of_pixel_inside /= subsampling_factor^2
                mask[i, j] = fraction_of_pixel_inside * α + (1 - fraction_of_pixel_inside) * 1.0

            end
        end
    end

    return mask

end



function create_imaging_bleach_mask(β::T, n_pixels::S, n_pad_pixels::S) where {T <: Real,S <: Integer}

    mask    = ones((n_pixels + 2 * n_pad_pixels, n_pixels + 2 * n_pad_pixels))
    mask[n_pad_pixels + 1:end - n_pad_pixels, n_pad_pixels + 1:end - n_pad_pixels] .= β

    return mask

end

function create_bleach_mask_legacy(α::T, γ::T, n_pixels::S, n_pad_pixels::S, bleach_region::AbstractROI{T}) where {T <: Real,S <: Integer}

    upsampling_factor = 15; # Needs to be a multiple of 3 due the 'box' method in imresize.

    # Create an upsampled bleach region to later downsample
    bounds              = get_bounds(bleach_region, γ)
    bleach_region_mask  = get_bleach_region_mask(bounds, bleach_region, α, upsampling_factor)

    # Smooth mask
    if γ > 0.0
        bleach_region_mask = filter_mask!(bleach_region_mask, γ, upsampling_factor)
    end

    # Downsample mask by the upsampling factor
    bleach_region_mask = downsample_mask!(bleach_region_mask, upsampling_factor) 

    mask    = ones((n_pixels + 2 * n_pad_pixels, n_pixels + 2 * n_pad_pixels))
    mask[n_pad_pixels + bounds.lb_x:n_pad_pixels + bounds.ub_x, n_pad_pixels + bounds.lb_y:n_pad_pixels + bounds.ub_y] .= bleach_region_mask

    return mask

end



function create_fourier_grid(n_pixels::T) where {T <: Integer}

    # Julia has no ndgrid or meshgrid function, out of principle
    # it would seem. Not "Julian" enough.
    x = range(-n_pixels ÷ 2, length=n_pixels)
    y = range(-n_pixels ÷ 2, length=n_pixels)

    # List comprehensions are very quick
    X = [j for i in x, j in y]
    Y = [i for i in x, j in y]

    X *= (2π / n_pixels)
    Y *= (2π / n_pixels)

    # Centre the transform
    ξ² = ifftshift(X.^2 + Y.^2)

    return ξ²
end



function meshgrid(x, y)

    # List comprehensions are very quick
    X = [i for i in x, j in y]
    Y = [j for i in x, j in y]

    return X, Y
end # function
