using RootFindGF
using Base.Test
using Nemo

function test_all()
    test_Γ()
    test_multi_eval()
    test_interpolation()
    test_resultant()
    test_SRA()
    test_flint()
end

K, x = FiniteField(2, 10, "x")
P, X = PolynomialRing(K, "X")

function test_Γ()
    Γ(map(i -> x^i, 0:9))
end

function test_multi_eval()
    list = map(i -> x^i, 0:400)
    f = X^400 + X^30 + x
    @test RootFindGF.multi_eval(f, list, 10) == [f(p) for p in list]
end

function test_interpolation()
    xs = map(i -> x^i, 0:400)
    f = X^400 + X^30 + x
    ys = map(f, xs)
    @test RootFindGF.interpolate(P, xs, ys) == f
end

function test_resultant()
    PP, Y = PolynomialRing(P, "Y")
    f = X^400 + X^30 + x
    @test RootFindGF.resultant(f, x) == resultant(compose(f, Y), X - Y^2 + x*Y)
end

function test_SRA()
    roots = [x^i for i in 0:300]
    pol = prod([X - r for r in roots])
    @test Set(roots) == Set(SRA(pol))
end

function test_flint()
    roots = [x^i for i in 0:300]
    pol = prod([X - r for r in roots])
    @test Set(roots) == Set(keys(flint_root_find(pol)))
end

test_all()
