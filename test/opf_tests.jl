# On a CUDA backend, MadNLP's defaults (SparseKKTSystem + MUMPS) are CPU-only and cannot
# assemble the KKT matrix from device arrays, so solve the condensed system with cuDSS on
# the GPU. On the CPU, pin the linear solver to Umfpack: MadNLP 0.10 changed the sparse
# default to MUMPS, which drives these OPFs into restoration, whereas Umfpack (the default
# through MadNLP 0.8, and what main uses) converges cleanly.
function exasolve(model, backend; kwargs...)
    opts = backend isa CUDABackend ?
        (; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver) :
        (; linear_solver = MadNLP.UmfpackSolver)
    return madnlp(model; opts..., kwargs...)
end

function test_case3(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)

    #Branches are encoded differently in solutions, so matches are hard coded
    vars_dict =  Dict("p" => p, "q" => q)
    for st_var in ["p", "q"]
        var = vars_dict[st_var]
        for i in 1:length(result_pm["solution"]["branch"])
            @test isapprox(Array(solution(result, var))[i], result_pm["solution"]["branch"][string(i)][string(st_var, "f")], atol = result.options.tol*100)
        end
    end
end

function test_static_case(result, result_pm, result_nlp_pm, pg, qg)
    @test result.status == result_nlp_pm.status
    @test isapprox(result.objective, result_pm["objective"], rtol = result.options.tol*100)
    for i in 1:length(result_pm["solution"]["gen"])
        @test isapprox(Array(solution(result, pg))[i], result_pm["solution"]["gen"][string(i)]["pg"], atol = result.options.tol*1000)
        @test isapprox(Array(solution(result, qg))[i], result_pm["solution"]["gen"][string(i)]["qg"], atol = result.options.tol*1000)
    end
end

function test_polar_voltage(result, result_pm, va, vm)
    for i in 1:length(result_pm["solution"]["bus"])
        @test isapprox(Array(solution(result, va))[i], result_pm["solution"]["bus"][string(i)]["va"], atol = result.options.tol*100)
        @test isapprox(Array(solution(result, vm))[i], result_pm["solution"]["bus"][string(i)]["vm"], rtol = result.options.tol*100)
    end
end

function test_rect_voltage(result, result_pm, vr, vim)
    for i in 1:length(result_pm["solution"]["bus"])
        @test isapprox(Array(solution(result, vr))[i], result_pm["solution"]["bus"][string(i)]["vr"], rtol = result.options.tol*100)
        @test isapprox(Array(solution(result, vim))[i], result_pm["solution"]["bus"][string(i)]["vi"], atol = result.options.tol*100)
    end
end

function test_case5(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)
end

function test_case14(result, result_pm, result_nlp_pm, pg, qg, p, q)
    test_static_case(result, result_pm, result_nlp_pm, pg, qg)
end

# Every configuration is checked by evaluating the model callbacks -- objective,
# constraints, gradient and Jacobian -- and comparing the Float32 model against the
# Float64 one.  The evaluation point is the model's own initial point rather than a
# solution, so this needs no solve, which is what makes it cheap enough to run over the
# whole cross-product.
function test_callbacks(m32, m64, backend)
    x1 = m64.meta.x0
    tol = 1e-4
    x2 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    x3 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    for x in [x1, x2, x3]
        o64 = NLPModelsJuMP.obj(m64, x)
        @test isfinite(o64)
        @test isapprox(NLPModelsJuMP.obj(m32, x), o64, rtol = tol)
        c64 = NLPModelsJuMP.cons(m64, x)
        @test all(isfinite, Array(c64))
        @test isapprox(NLPModelsJuMP.cons(m32, x), c64, rtol = tol)
        if backend != CUDABackend()
            @test isapprox(NLPModelsJuMP.grad(m32, x), NLPModelsJuMP.grad(m64, x), rtol = tol)
            @test isapprox(NLPModelsJuMP.jac(m32, x), NLPModelsJuMP.jac(m64, x), rtol = tol)
        end
    end
end

function test_mp_case(result, true_sol)
    @test result.status == MadNLP.SOLVE_SUCCEEDED || result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
    @test isapprox(result.objective, true_sol, rtol = result.options.tol*100)
end

# GOC3 is a smoke test of the parser and the model constructor.  Solving it cost 32 of the
# 47 min this section took in CI, for one iteration of a 139502-variable problem, so build
# the model and evaluate its callbacks instead of solving.
# The model may not depend on uid spellings. Rename every uid so the digits
# the retired suffix rule would have read run in reverse of document order;
# the model built from the renamed pair must be the one built from the
# original. This is the regression for the misindexing that rule caused on
# files that do not follow the competition uid convention.
function sc_uid_invariance_tests(filename, backend, T)
    case_path = filename * ".json"
    uc_path = filename * "_solution.json"
    case = JSON.parsefile(case_path)
    uc = JSON.parsefile(uc_path)
    uids = String[]
    for section in values(case["network"])
        section isa Vector || continue
        for item in section
            item isa Dict && haskey(item, "uid") && push!(uids, String(item["uid"]))
        end
    end
    n = length(uids)
    mapping = Dict(uid => "renamed $(n - i) ($uid)" for (i, uid) in enumerate(uids))
    rename(x::Dict) = Dict(k => rename(v) for (k, v) in x)
    rename(x::Vector) = [rename(v) for v in x]
    rename(x::String) = get(mapping, x, x)
    rename(x) = x
    dir = mktempdir()
    renamed_case = joinpath(dir, "case.json")
    renamed_uc = joinpath(dir, "case_solution.json")
    open(renamed_case, "w") do io
        JSON.print(io, rename(case))
    end
    open(renamed_uc, "w") do io
        JSON.print(io, rename(uc))
    end
    m0, _, vars0, lengths0, _ = ExaModelsPower.goc3_model(case_path, uc_path; backend = backend, T = T)
    m1, _, vars1, lengths1, _ = ExaModelsPower.goc3_model(renamed_case, renamed_uc; backend = backend, T = T)
    x0 = m0.meta.x0
    @test m1.meta.nvar == m0.meta.nvar
    @test NLPModelsJuMP.obj(m1, x0) ≈ NLPModelsJuMP.obj(m0, x0)
    @test Array(NLPModelsJuMP.cons(m1, x0)) ≈ Array(NLPModelsJuMP.cons(m0, x0))

    # The solution writer maps every row back to a variable index through the
    # instance's uid tables. Write the same variable vector against both
    # documents; each written row must carry the values its original carries.
    x = collect(x0)
    S = typeof(x)
    result = ExaModels.SolverCore.GenericExecutionStats{eltype(x),S,S,Any}(solution = x)
    written = joinpath(dir, "written.json")
    written_renamed = joinpath(dir, "written_renamed.json")
    ExaModelsPower.save_go3_solution(case_path, uc_path, written, result, vars0, lengths0)
    ExaModelsPower.save_go3_solution(renamed_case, renamed_uc, written_renamed, result, vars1, lengths1)
    w0 = JSON.parsefile(written)["time_series_output"]
    w1 = JSON.parsefile(written_renamed)["time_series_output"]
    @test Set(keys(w0)) == Set(keys(w1))
    for (section, rows) in w0
        by_uid = Dict(String(r["uid"]) => r for r in w1[section])
        @test length(by_uid) == length(rows)
        for row in rows
            renamed_row = by_uid[rename(String(row["uid"]))]
            for (field, value) in row
                @test renamed_row[field] == rename(value)
            end
        end
    end
end

function sc_tests(filename, backend, T)
    uc_filename = filename*"_solution.json"
    filename = filename*".json"
    model, cons, vars, lengths, sc_data_array = ExaModelsPower.goc3_model(filename, uc_filename; backend=backend, T=T)
    x0 = model.meta.x0
    @test isfinite(NLPModelsJuMP.obj(model, x0))
    @test all(isfinite, Array(NLPModelsJuMP.cons(model, x0)))
end

function test_dcopf_case(result, result_pm, pg, pf)
    @test result.status == MadNLP.SOLVE_SUCCEEDED || result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
    @test isapprox(result.objective, result_pm["objective"], rtol = result.options.tol*100)
    for i in 1:length(result_pm["solution"]["gen"])
        @test isapprox(Array(solution(result, pg))[i], result_pm["solution"]["gen"][string(i)]["pg"], atol = result.options.tol*1000)
    end
    for i in 1:length(result_pm["solution"]["branch"])
        @test isapprox(Array(solution(result, pf))[i], result_pm["solution"]["branch"][string(i)]["pf"], atol = result.options.tol*1000)
    end
end
