module RootFindGF
using Nemo

export Γ, SRA, flint_root_find

include("tree.jl")
include("c_libs.jl")

"""
Compute the γ_i,j = L_i(v_j) from a basis.
"""
function Γ{T<:Nemo.FinFieldElem}(basis::AbstractArray{T,1}, stop::Integer=0)
    K = parent(basis[1])
    p = characteristic(K)
    n = degree(K)
    m = max(n - stop, 1)
    n ≠ length(basis) && error("Basis has wrong dimension")
    
    # Uses the plain recursive definition of L_i.
    # Complexity is evidently O(n² log p) mulitpilication in K.
    Γ = zeros(K, m, n)
    Γ[1,:] = basis
    for i in 2:m
        α = Γ[i-1, i-1]^(p-1)
        for (j, v) in enumerate(basis)
            γ = Γ[i-1, j]
            Γ[i, j] = γ^p - α * γ
        end
    end
    Γ
end

"""
SRA Algorithm

We switch to exahustive search one step before the root space becomes
smaller than the degree of `f`, or when it is smaller than `thresh`.
Equivalently, the number of resultant steps is such that

    p^(n - steps - 1) ≤ d < p^(n - steps)

or

    p^(n - steps) ≤ thresh < p^(n - steps + 1)

whichever is smallest.
"""
function SRA{T<:Nemo.FinFieldElem}(f::PolyElem{T}, thresh::Integer=0)
    K = base_ring(f)
    p = Int(characteristic(K))
    n = degree(K)
    d = degree(f)
    d < order(K) || error("Polynomial degree too large")

    # This condition also ensures that if we need to call
    # `resultant`, then d is small enough for the algorithm
    # to run.
    steps = Int(n - max(floor(log(p, d)) + 1, floor(log(p, thresh))))

    z = Nemo.gen(K)
    return _SRA(f, Γ(T[z^i for i in 0:n-1], n - steps - 1))
end

"""
Internal SRA function, using precomputed matrix Γ.
"""
function _SRA{T<:Nemo.FinFieldElem}(f::PolyElem{T}, Γ::AbstractArray{T,2}, stop::Integer=0)
    K = base_ring(f)
    p = Int(characteristic(K))
    n = degree(K)
    m = max(size(Γ, 1) - stop, 1)
    d = degree(f)
    
    # Compute the first m-1 resultants
    fs = Array{PolyElem{T}}(m)
    fs[1] = f
    for i in 2:m
        fs[i] = resultant(fs[i-1], Γ[i-1, i-1])
    end

    # Intialize the root list with the full vector space of
    # dimension n-m
    vec = Array{Int}(n)
    roots = Array{NTuple{n,Int}}(p^(n-m))
    for i in 0:length(roots)-1
        roots[i+1] = tuple(reverse!(digits!(vec, i, p))...)
    end

    valroots = Array{T}(d)
    for i in m:-1:1
        roots = multi_roots!(valroots, fs[i], roots, squeeze(Γ[i,:], 1), i, Γ[i,i])
    end

    return valroots
end


"""
Computes the resultant

    Res_x(f(x), y - x^p + β^(p-1)x)
"""
function resultant{T<:Nemo.FinFieldElem}(f::PolyElem{T}, β::T)
    P = parent(f)
    K = base_ring(f)
    p = Int(characteristic(K))
    d = degree(f)
    parent(β) != K && error("Inputs have different base fields")
    d ≥ order(K) // p && error("Polynomial degree too large")

    α = β^(p-1)
    # Select enough points avoiding β
    points = avoid_β(β, d+1)
    evaluated = multi_eval_β(f, points, β)
    # Compute products by row
    evals = squeeze(reducedim(*, evaluated, 2, K(1)), 2)
    # Interpolate at the points p_i^p - α p_i with values computed above
    interp = interpolate(P, map(x -> x^p - α*x, points), evals)
    return (-1)^(d % 2) * interp
end

"""
Multipoint evaluation of `f` at (p_i + cβ) for p_i ∈ points and c ∈ GF(p)
"""
function multi_eval_β{T<:Nemo.FinFieldElem}(f::PolyElem{T}, points::AbstractArray{T,1}, β::T)
    K = base_ring(f)
    p = Int(characteristic(K))

    # Compute the span of β
    shifts = cumsum(fill(β, 1, p), 2)
    # Compute the matrix (p_i + cβ) with p_i in points and c in GF(p)
    expanded = broadcast(+, points, shifts)
    # Evaluate f at each point in the matrix above
    return mapslices(col -> multi_eval(f, col), expanded, 1)
end

"""
Helper function for SRA
"""
function multi_roots!{T<:Nemo.FinFieldElem,n}(vals::Array{T,1},
                                                 f::PolyElem{T}, roots::Array{NTuple{n,Int},1},
                                                 basis::AbstractArray{T,1}, index::Integer, β::T)
    p = Int(characteristic(parent(β)))
    # compute the dot products of each root
    points = map(r -> sum(map(*,r,basis)), roots)
    # do multi-point evaluation
    evals = multi_eval_β(f, points, β)
    # look for zeros
    zeros = map(i -> ind2sub(size(evals), i), find(Nemo.iszero, evals))
    newroots = Array{NTuple{n,Int}}(length(zeros))
    for (i, (r,c)) in enumerate(zeros)
        update(tup, c) = i -> i == index ? c % p : tup[i]
        newroots[i] = ntuple(update(roots[r], c), n)
        vals[i] = points[r] + c*β
    end
    return newroots
end

"""
Returns a list of `d` elements whose span does not contain `b`.
"""
function avoid_β{T<:Nemo.FinFieldElem}(β::T, d::Int)
    K = parent(β)
    n = degree(K)
    p = Int(characteristic(K))
    z = gen(K)

    # We pick the first d elements in the span of
    #     1, ..., z^(skip-1), z^(skip+1), ..., z^(n-1)
    # where z^skip is such that its coefficient in β is non-zero.
    skip = findfirst([coeff(β,i) for i in 0:n-1]) - 1

    points = Array{T}(d)
    points[1] = K(0)
    for i in 2:d
        # To iterate the span, we use a neat Gray-code trick
        val(n) = n % p == 0 ? val(n ÷ p) + 1 : 0
        exp = let e = val(i-1); e < skip ? e : e+1 end
        points[i] = points[i-1] + z^exp
    end

    return points
end

end
