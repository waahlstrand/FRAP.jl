module FRAP
    
    # include("device.jl")
    include("concentration.jl")
    include("mask.jl")
    include("params.jl")
    include("core.jl")
    include("fit.jl")
    include("recovery.jl")

    # export cpu, gpu
    export Signal
    export run
    export concentration
    export ExperimentParams, BathParams
    export AbstractROI, CircleROI, RectangleROI
    export diffuse!, evolve!
    export create_fourier_grid
    export create_imaging_bleach_mask, create_bleach_mask, create_bleach_region
    export recovery_curve
    export residual

end
