abstract Tree{T}

type Root{T} <: Tree{T}
end

type BinNode{T} <: Tree{T}
    val::T
    left::Tree{T}
    right::Tree{T}
    parent::Tree{T}
end

type NAryNode{T} <: Tree{T}
    val::T
    children::Array{T,1}
    parent::Tree{T}
end

type Leaf{T} <: Tree{T}
    val::T
    parent::Tree{T}
end


"""
Compute the binary subproduct tree of `leaves`
"""
function subproduct_tree{U}(leaves::AbstractArray{U,1})
    n = length(leaves)
    n == 0 && error("Must have at least one leaf")
    if n == 1
        return Leaf{U}(leaves[1], Root{U}())
    else
        m = n ÷ 2
        left  = subproduct_tree(sub(leaves, 1:m))
        right = subproduct_tree(sub(leaves, m+1:n))
        new = BinNode{U}(left.val * right.val, left, right, Root{U}())
        left.parent = right.parent = new
        return new
    end
end

"""
Compute the multi-point evaluation of `f` at `points`.

`thresh` is a threshold to switch between naive evaluation and subproduct-tree
based evaluation. Default is 10 (1 == no naive evaluation).
"""
function multi_eval{T<:Nemo.FieldElem}(f::PolyElem{T}, points::AbstractArray{T,1}, thresh=10)
    X = Nemo.gen(parent(f))
    # Reshape the array in slices of length `thresh`
    n = length(points)
    view = map(i -> sub(points, i:min(i+thresh-1, n)), 1:thresh:n)
    # Collect the array per slice, using naive multiplication
    gather(ps) = mapreduce(c -> (X-c), *, ps)
    leaves = map(gather, view)
    # Construct subproduct tree
    reduced = multi_eval_tree(f, subproduct_tree(leaves))
    # Apply naive evaluation to each slice
    eval(g, ps) = map(g, ps)
    return vcat(map(eval, reduced, view)...)
end

"""
Compute the multi-point evaluation of `f` at the subproduct tree `tree`
"""
function multi_eval_tree{T<:Nemo.PolyElem}(f::T, tree::Tree{T})
    error("Pass a BinNode or Leaf instance.")
end
function multi_eval_tree{T<:Nemo.PolyElem}(f::T, tree::Leaf{T})
    return [f % tree.val]
end
function multi_eval_tree{T<:Nemo.PolyElem}(f::T, tree::BinNode{T})
    g = f % tree.val
    return [multi_eval_tree(g, tree.left);
            multi_eval_tree(g, tree.right)]
end

"""
Fast polynomial interpolation
"""
function interpolate{T<:Nemo.FieldElem}(P::Nemo.Ring, xs::AbstractArray{T,1},
                                        ys::AbstractArray{T,1}, thresh=5)
    length(xs) == length(ys) || error("Vectors must have same length")
    base_ring(P) == parent(xs[1]) == parent(ys[1]) || error("Must belong to the same field")
    
    X = Nemo.gen(P)
    # Reshape the array in slices of length `thresh`
    n = length(xs)
    view(points) = map(i -> sub(points, i:min(i+thresh-1, n)), 1:thresh:n)
    xview, yview = view(xs), view(ys)
    # Collect the array per slice, using naive multiplication
    gather(ps) = mapreduce(c -> (X-c), *, ps)
    leaves = map(gather, xview)
    # Construct subproduct tree
    tree = subproduct_tree(leaves)
    # Apply naive interpolation to each slice
    denominators = multi_eval_tree(derivative(tree.val), tree)
    eval(f, g, xs, ys) = mapreduce((t) -> (f ÷ (X-t[1])) * t[2]//g(t[1]), +, zip(xs, ys))
    floor = map(eval, leaves, denominators, xview, yview)
    # Recursive interpolation
    return interpolate_tree(floor, tree, IntWrapper(0))
end

type IntWrapper
    c::Int
end
function interpolate_tree{T<:Nemo.PolyElem}(floor::Array{T,1}, tree::Leaf{T},
                                            count::IntWrapper)
    count.c += 1
    return floor[count.c]
end
function interpolate_tree{T<:Nemo.PolyElem}(floor::Array{T,1}, tree::BinNode{T},
                                            count::IntWrapper)
    left = interpolate_tree(floor, tree.left, count)
    right = interpolate_tree(floor, tree.right, count)
    return left*tree.right.val + right*tree.left.val
end
