module FRAP
    
    include("concentration.jl")
    include("mask.jl")
    include("params.jl")
    include("core.jl")
    include("fit.jl")
    include("recovery.jl")

    export signal
    export concentration
    export ExperimentParams, BathParams
    export AbstractROI, CircleROI, RectangleROI
    export diffuse!, evolve!
    export fourier_grid
    export imaging_mask, bleaching_mask, region_of_interest
    export recovery_curve
    export residual

end
