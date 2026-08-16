# Minimal N-1 SCOPF regression test.
#
# Mirrors examples/scopf.jl on the small case9 network with its 2 single-line
# contingencies (data/case9.Ctgs): it solves the SAME N-1 SCOPF three ways and
# checks they all agree on the objective AND on the full base-case and
# per-scenario generator dispatch (the two formulations are mathematically
# identical, so a disagreement is a real bug, not a tolerance artifact):
#
# EVERY solve here is CONDENSED, and that is load-bearing rather than incidental.
# The Schur path is inherently condensed (RelaxEquality with bound_relax_factor =
# tol), so a monolithic solve left on the DEFAULT sparse KKT optimizes the
# hard-equality problem instead and the two disagree by the relaxation, not by a
# bug. Measured on case9/K=2/CPU/Float64: matching the relaxation gives objective
# gap 0.0 and dispatch gap 2.12e-13, where leaving `:single` hard gives 13.43 and
# 2.12e-3 -- and that 13.43 is ~2*tol of equality slack, so it scales WITH tol
# (1.4e-3 at tol=1e-8) and cannot be tuned away by tightening.
#
#   * CPU :single    — scopf_model            (monolithic ExaModel, condensed KKT + CHOLMOD)
#   * CPU :twostage  — scopf_twostage_model    (Schur complement KKT + MUMPS)
#   * GPU :twostage  — scopf_twostage_model    (Schur complement KKT + cuDSS), skipped
#                                               when no CUDA GPU is present
#
# CPU :single is the trusted reference. GPU :single is deliberately NOT tested: the
# condensed-KKT GPU path has been observed to report success at a point that does not
# satisfy power balance, so it cannot serve as the thing everything else is checked
# against.

# Build the SchurComplementCondensedKKTSystem kkt_options from the model's post_solve_info
# tags (MadNLP can't auto-detect ExaModels' interleaved design/scenario tag names).
function scopf_schur_kkt_options(info)
    return Dict{Symbol,Any}(
        :schur_ns => info.ns, :schur_nv => info.nv, :schur_nd => info.nd, :schur_nc => info.nc,
        :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen,
    )
end

# Solve the monolithic scopf_model. `vars.pg` is ngen × (K+1): column 1 is the base
# case, columns 2..K+1 are the K contingencies. Returns (result, pg).
function solve_scopf_single(case, contingencies, backend)
    model, vars, _ = scopf_model(case, contingencies; backend = backend)
    opts = backend isa CUDABackend ?
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver) :
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver)
    result = madnlp(model; tol = 1.0e-4, print_level = MadNLP.ERROR, opts...)
    return result, Array(solution(result, vars.pg))
end

# Solve the two-stage scopf_twostage_model via the Schur complement KKT system (MUMPS
# on CPU, cuDSS on GPU). `vars.pg0` is the base dispatch (ngen); `vars.pgk` the per-
# scenario dispatch (ngen × K). Returns (result, pg0, pgk).
function solve_scopf_twostage(case, contingencies, backend; inertia = MadNLP.InertiaBased)
    model, vars, _, info = scopf_twostage_model(case, contingencies; backend = backend)
    lin = backend isa CUDABackend ? MadNLPGPU.CUDSSSolver : MadNLP.MumpsSolver
    result = madnlp(model;
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SchurComplementCondensedKKTSystem,
        linear_solver = lin,
        kkt_options = scopf_schur_kkt_options(info),
        inertia_correction_method = inertia,
        tol = 1.0e-4, print_level = MadNLP.ERROR,
    )
    return result, Array(solution(result, vars.pg0)), Array(solution(result, vars.pgk))
end

converged(r) = r.status == MadNLP.SOLVE_SUCCEEDED || r.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

# Assert a two-stage solve agrees with the reference single-case dispatch. Both
# layouts are generator-fast / scenario-slow, so compare flattened (robust to whether
# `solution` hands back a matrix or a flat vector for the per-scenario variable).
function test_scopf_agrees(r, pg0, pgk, r_ref, pg0_ref, pgk_ref)
    @test converged(r)
    @test isapprox(r.objective, r_ref.objective, rtol = 1.0e-3)
    @test isapprox(vec(pg0), vec(pg0_ref), atol = 1.0e-3)   # base-case dispatch
    @test isapprox(vec(pgk), vec(pgk_ref), atol = 1.0e-3)   # per-scenario dispatch
end

# The DC formulation of the same N-1 problem. `scopf_twostage_model` is AC-only,
# so there is no second DC solve to check against — this checks the MECHANISM
# the DC outage rests on instead.
#
# In AC an outage zeroes the admittance coefficients `c1..c8`; in DC those play
# no part, and the outage is a zero written into the susceptance `bs` that
# `_scopf_narrow` carries on each branch row. If that zero failed to reach the
# flow equation the branch would keep conducting, and the model would still
# build, still converge, and still report a plausible cost — so the flow itself
# is what is asserted.
#
# Armed on both sides: the outaged line must carry real flow in the BASE case
# (or an all-zero solution would pass), and some other line must carry real flow
# in the SAME scenario (or a dead scenario would pass).
function test_scopf_dc(case, contingencies)
    model, vars, _ = scopf_model(case, contingencies; form = DC())
    result = madnlp(model; tol = 1.0e-8, print_level = MadNLP.ERROR)
    @test converged(result)

    pf = Array(solution(result, vars.pf))        # nbranch x (K+1)
    @test size(pf, 2) == length(contingencies) + 1

    for (k, ct) in enumerate(contingencies)
        l = ct.idx
        c = k + 1                                 # scenario 1 is the base case
        @test isapprox(pf[l, c], 0.0, atol = 1.0e-8)                 # the outage happened
        @test abs(pf[l, 1]) > 1.0e-4                                 # on a line that was carrying
        @test maximum(abs, pf[setdiff(1:size(pf, 1), (l,)), c]) > 1.0e-4  # in a live scenario
    end
end

function scopf_tests(; cpu = true, gpu = CUDA.has_cuda_gpu())
    (cpu || gpu) || return nothing

    case = joinpath(@__DIR__, "..", "data", "case9.m")
    # Each line of case9.Ctgs is a 1-based branch index to outage (just like the example).
    ctg_idxs = parse.(Int, filter(!isempty, strip.(readlines(joinpath(@__DIR__, "..", "data", "case9.Ctgs")))))
    contingencies = [(type = :branch, idx = l) for l in ctg_idxs]
    K = length(contingencies)

    @testset "SCOPF case9 N-1 (K=$K)" begin
        # CPU :single is the reference solution. The GPU comparison needs it too,
        # so it is not gated on `cpu`.
        r_single, pg_single = solve_scopf_single(case, contingencies, nothing)
        @test converged(r_single)
        pg0_ref = pg_single[:, 1]
        pgk_ref = pg_single[:, 2:end]

        if cpu
            @testset "CPU two-stage matches single" begin
                r, pg0, pgk = solve_scopf_twostage(case, contingencies, nothing)
                test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
            end

            @testset "DC outage zeroes the outaged line" begin
                test_scopf_dc(case, contingencies)
            end
        end

        if gpu
            @testset "GPU two-stage matches single" begin
                r, pg0, pgk = solve_scopf_twostage(case, contingencies, CUDABackend())
                test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
            end
        end
    end
    return nothing
end
