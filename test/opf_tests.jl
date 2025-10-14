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
    @info "ppolar"
    @info result_pm
    for i in 1:length(result_pm["solution"]["bus"])
        @test isapprox(Array(solution(result, va))[i], result_pm["solution"]["bus"][string(i)]["va"], atol = result.options.tol*100)
        @test isapprox(Array(solution(result, vm))[i], result_pm["solution"]["bus"][string(i)]["vm"], rtol = result.options.tol*100)
    end
end

function test_rect_voltage(result, result_pm, vr, vim)
    @info "rrect"
    @info result_pm
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

function test_float32(m, m64, result, backend)
    x1 = result.solution
    tol = 2.71828^(log(result.options.tol)/2)
    x2 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    x3 = x1 .* (1 .+ 0.01 .* (2 .* rand(size(x1)) .- 1))
    for x in [x1, x2, x3]
        @test isapprox(NLPModelsJuMP.obj(m, x), NLPModelsJuMP.obj(m64, x), rtol = tol)
        @test isapprox(NLPModelsJuMP.cons(m, x), NLPModelsJuMP.cons(m64, x), rtol = tol)
        if backend != CUDABackend()
            @test isapprox(NLPModelsJuMP.grad(m, x), NLPModelsJuMP.grad(m64, x), rtol = tol)
            @test isapprox(NLPModelsJuMP.jac(m, x), NLPModelsJuMP.jac(m64, x), rtol = tol)
        end
    end
end

function test_mp_case(result, true_sol)
    @test result.status == MadNLP.SOLVE_SUCCEEDED || result.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
    @test isapprox(result.objective, true_sol, rtol = result.options.tol*100)
end

function sc_tests(filename, backend, T)
    uc_filename = filename*"_solution.json"
    filename = filename*".json"
    model, cons, vars, lengths, sc_data_array = ExaModelsPower.goc3_model(filename, uc_filename; backend=backend, T=T)
    result = madnlp(model; max_iter=5, tol=1e-2)
end