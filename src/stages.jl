abstract type BleachStage end

struct Prebleach{T<:Real, S<:Integer}

    masks::Array{Array{T, 2}, 1}
    frames::UnitRange{S}

end