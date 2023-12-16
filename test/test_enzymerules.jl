using VLBISkyModels
using Enzyme


u = rand(10_000) .- 0.5
v = rand(10_000) .- 0.5

buffer = IntensityMap(zeros(4, 4), 1.0, 1.0)
cache = create_cache(NFFTAlg(u, v), buffer, DeltaPulse())


f = let p = cache.plan
        x->sum(abs2, VLBISkyModels.nuft(p, x))
end

x = rand(4, 4)
dx = zero(x)
f(x)
foo(x, p) = sum(abs2, VLBISkyModels.nuft(p, x))
Enzyme.autodiff(Reverse, foo, Active, Duplicated(x, fill!(dx, 0.0)), Const(cache.plan))

out = zeros(ComplexF64, length(u))
using Zygote

using EnzymeTestUtils

for Tout in (Enzyme.Duplicated, Enzyme.BatchDuplicated),
    TA in (Enzyme.Const,),
    Tb in (Enzyme.Duplicated, Enzyme.BatchDuplicated)

    EnzymeTestUtils.are_activities_compatible(Tout, TA, Tb) || continue

    EnzymeTestUtils.test_reverse(VLBISkyModels._nuft!, Const, (out, Tout), (cache.plan, TA), (x, Tb))
  end


using Zygote

gz, = Zygote.gradient(x->foo(x, cache.plan), x)
