function concentration(c₀::T, ϕₘ::T, dims::Tuple{S,Vararg{S}}) where {T<:Real, S<:Integer}

    mobile = c₀ * ϕₘ * ones(T, dims)
    immobile = c₀ * (1 - ϕₘ) * ones(T, dims)

    return mobile, immobile

end