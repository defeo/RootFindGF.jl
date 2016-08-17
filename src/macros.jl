"""
Macro to extract row `i` from matrix `M`, for compatibility across Julia releasesq
"""
macro row(M, i)
    if VERSION < @v_str("0.5-")
        return :( squeeze($M[$i,:], 1) )
    else
        return :( $M[$i,:] )
    end
end
