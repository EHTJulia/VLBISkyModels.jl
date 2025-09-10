using EnzymeTestUtils
using NonuniformFFTs

# Hack so that it doesn't try to compare nfft plans since they are const and have a FF pointer
function EnzymeTestUtils.test_approx(
        ::VLBISkyModels.NFFT.NFFTPlan,
        ::VLBISkyModels.NFFT.NFFTPlan, args...; kwargs...
    )
    return true
end

function EnzymeTestUtils.test_approx(
        ::PlanNUFFT,
        ::PlanNUFFT, args...; kwargs...
    )
    return true
end


@testset "nfft Enzyme rules" begin
    out = zeros(ComplexF64, 10)
    n = 16
    k = rand(2, 10) .- 0.5
    plan = VLBISkyModels.NFFT.plan_nfft(k, (n, n))
    b = rand(n, n)
    for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
        are_activities_compatible(Const, Tret, Tb) || continue
        test_reverse(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    end
    for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
        are_activities_compatible(Const, Tret, Tb) || continue
        test_forward(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    end
end

@testset "FINUFFT Enzyme rules" begin
    g = imagepixels(10.0, 10.0, 16, 16)
    U = randn(64)
    V = randn(64)
    guv = UnstructuredDomain((; U, V))
    gfi = FourierDualDomain(g, guv, VLBISkyModels.FINUFFTAlg())

    plan = VLBISkyModels.forward_plan(gfi).plan
    b = zeros(size(g))
    out = zeros(ComplexF64, length(U))
    for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
        are_activities_compatible(Const, Tret, Tb) || continue
        test_reverse(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    end
    # TODO Why is this not working?
    # for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
    #     are_activities_compatible(Const, Tret, Tb) || continue
    #     test_forward(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    # end
end

@testset "NonuniformFFTs Enzyme rules" begin
    g = imagepixels(10.0, 10.0, 16, 16)
    U = randn(64)
    V = randn(64)
    guv = UnstructuredDomain((; U, V))
    gnu = FourierDualDomain(g, guv, VLBISkyModels.NonuniformFFTsAlg())

    plan = VLBISkyModels.forward_plan(gnu).plan
    b = zeros(size(g))
    out = zeros(ComplexF64, length(U))
    for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
        are_activities_compatible(Const, Tret, Tb) || continue
        test_reverse(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    end
    # TODO Why is this not working?
    # for Tret in (Duplicated, BatchDuplicated), Tb in (Duplicated, BatchDuplicated)
    #     are_activities_compatible(Const, Tret, Tb) || continue
    #     test_forward(VLBISkyModels._jlnuft!, Const, (out, Tret), (plan, Const), (b, Tb))
    # end
end
