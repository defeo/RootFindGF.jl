
###############################################################################
#
# Flint
#
###############################################################################

"""
Flint's own root-finding algorithm, as called via *_poly_factor_equal_deg
"""
function flint_root_find(x::fq_nmod_poly)
   R = parent(x)
   F = base_ring(R)
   fac = Nemo.fq_nmod_poly_factor(F)
   ccall((:fq_nmod_poly_factor_equal_deg, :libflint), Void, 
         (Ptr{Nemo.fq_nmod_poly_factor}, Ptr{fq_nmod_poly}, Int,
         Ptr{FqNmodFiniteField}), &fac, &x, 1, &F)
   res = Dict{fq_nmod, Int}()
   for i in 1:fac.num
      f = R()
      ccall((:fq_nmod_poly_factor_get_poly, :libflint), Void,
            (Ptr{fq_nmod_poly}, Ptr{Nemo.fq_nmod_poly_factor}, Int,
            Ptr{FqNmodFiniteField}), &f, &fac, i-1, &F)
      d = unsafe_load(fac.exp,i)
      res[-coeff(f,0)] = d
   end
   return res
end
