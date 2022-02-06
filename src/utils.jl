function device(d)

    # Select device
    if CUDA.functional() && d == "gpu"
        cpu_or_gpu = function(x::AbstractArray) 
            return CuArray(x)
    end
    elseif (d == "cpu")
        cpu_or_gpu = function(x::AbstractArray) 
                return x
        end    
    else 
        @warn "Invalid device; falling back on cpu."
        cpu_or_gpu = function(x::AbstractArray) 
            return x
        end    
    end

    return cpu_or_gpu

end
