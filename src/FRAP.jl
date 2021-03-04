
module FRAP
    include("concentration.jl")
    include("mask.jl")
    include("params.jl")
    include("core.jl")

    export run
    export concentration
    export Params, ExperimentParams, BathParams
    export AbstractROI, CircleROI, RectangleROI
    export diffuse!, evolve!
    export create_fourier_grid
    export create_imaging_bleach_mask, create_bleach_mask, create_bleach_region

end
