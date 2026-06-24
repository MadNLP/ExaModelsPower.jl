# Minimal N-1 SCOPF regression test.
#
# Mirrors examples/scopf.jl on the small case9 network with its 2 single-line
# contingencies (data/case9.Ctgs): it solves the SAME hard-constrained N-1 SCOPF
# three ways and checks they all agree on the objective AND on the full base-case
# and per-scenario generator dispatch (the two formulations are mathematically
# identical, so a disagreement is a real bug, not a tolerance artifact):
#
#   * CPU :single    — scopf_model            (monolithic ExaModel, default sparse KKT)
#   * CPU :twostage  — scopf_twostage_model    (Schur complement KKT + MUMPS)
#   * GPU :twostage  — scopf_twostage_model    (Schur complement KKT + cuDSS), skipped
#                                               when no CUDA GPU is present
#
# CPU :single is the trusted reference. GPU :single is deliberately NOT tested: the
# condensed-KKT GPU path can report success at a power-balance-infeasible point (see
# SCOPF-GPU-TWOSTAGE-FINDINGS.md / CLAUDE.md), so it is not a trustworthy reference.

# Build the SchurComplementKKTSystem kkt_options from the model's post_solve_info
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
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver) : (;)
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
        kkt_system = MadNLP.SchurComplementKKTSystem,
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

function scopf_tests()
    case = joinpath(@__DIR__, "..", "data", "case9.m")
    # Each line of case9.Ctgs is a 1-based branch index to outage (just like the example).
    ctg_idxs = parse.(Int, filter(!isempty, strip.(readlines(joinpath(@__DIR__, "..", "data", "case9.Ctgs")))))
    contingencies = [(type = :branch, idx = l) for l in ctg_idxs]
    K = length(contingencies)

    @testset "SCOPF case9 N-1 (K=$K)" begin
        # CPU :single is the reference solution.
        r_single, pg_single = solve_scopf_single(case, contingencies, nothing)
        @test converged(r_single)
        pg0_ref = pg_single[:, 1]
        pgk_ref = pg_single[:, 2:end]

        @testset "CPU two-stage matches single" begin
            r, pg0, pgk = solve_scopf_twostage(case, contingencies, nothing)
            test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
        end

        if CUDA.has_cuda_gpu()
            @testset "GPU two-stage matches single" begin
                r, pg0, pgk = solve_scopf_twostage(case, contingencies, CUDABackend())
                test_scopf_agrees(r, pg0, pgk, r_single, pg0_ref, pgk_ref)
            end
        end
    end
end
