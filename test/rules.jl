using EnzymeTestUtils

# Hack so that it doesn't try to compare nfft plans since they are const and have a FF pointer
EnzymeTestUtils.test_approx(::VLBISkyModels.NFFT.NFFTPlan, ::VLBISkyModels.NFFT.NFFTPlan, args...; kwargs...) = true

@testset "nfft reverse rule" begin
    out = zeros(ComplexF64, 10)
    n = 16
    k = rand(2,10) .- 0.5
    plan = VLBISkyModels.NFFT.plan_nfft(k, (n,n))
    b = rand(n, n)
    for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
        are_activities_compatible(Const, Tret, Tb) || continue
        test_reverse(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    end
end