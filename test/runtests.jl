using Test, ExaModelsPower, MadNLP, MadNLPGPU, KernelAbstractions, CUDA, PowerModels, Ipopt, JuMP, ExaModels, NLPModelsJuMP

include("opf_tests.jl")

const CONFIGS = [
    (Float64, nothing),
    (Float64, CPU()),
    (Float32, nothing),
    (Float32, CPU()),
]

if CUDA.has_cuda_gpu()
    push!(
        CONFIGS,
        (Float32, CUDABackend()),
    )
    push!(
        CONFIGS,
        (Float64, CUDABackend()),
    )
end

test_cases = [("../data/pglib_opf_case3_lmbd.m", "case3", test_case3),
              ("../data/pglib_opf_case5_pjm.m", "case5", test_case5),
              ("../data/pglib_opf_case14_ieee.m", "case14", test_case14)]

#MP
#MP solutions hard coded based on solutions computer 4/10/2025 on CPU with 1e-8 tol
#Curve = [1, .9, .8, .95, 1]
true_sol_case3_curve = 25384.366465
true_sol_case3_pregen = 29049.351564
true_sol_case5_curve = 78491.04247
true_sol_case5_pregen = 87816.396884
#W storage
true_sol_case3_curve_stor = 25358.8275
true_sol_case3_curve_stor_func = 25352.57 
true_sol_case3_pregen_stor = 29023.691
true_sol_case3_pregen_stor_func = 29019.32 
true_sol_case5_curve_stor = 68782.0125
true_sol_case5_curve_stor_func = 69271.9 
true_sol_case5_pregen_stor = 79640.085
true_sol_case5_pregen_stor_func = 79630.4 
mp_test_cases = [("../data/pglib_opf_case3_lmbd.m", "case3", "../data/case3_5split.Pd", "../data/case3_5split.Qd", true_sol_case3_curve, true_sol_case3_pregen),
                 ("../data/pglib_opf_case5_pjm.m", "case5", "../data/case5_5split.Pd", "../data/case5_5split.Qd", true_sol_case5_curve, true_sol_case5_pregen)]

mp_stor_test_cases = [("../data/pglib_opf_case3_lmbd_mod.m", "case3", "../data/case3_5split.Pd", "../data/case3_5split.Qd",
                        true_sol_case3_curve_stor, true_sol_case3_curve_stor_func, true_sol_case3_pregen_stor, true_sol_case3_pregen_stor_func),
                        ("../data/pglib_opf_case5_pjm_mod.m", "case5", "../data/case5_5split.Pd", "../data/case5_5split.Qd",
                        true_sol_case5_curve_stor, true_sol_case5_curve_stor_func, true_sol_case5_pregen_stor, true_sol_case5_pregen_stor_func)]

static_forms = [("rect", :rect, ACRPowerModel, test_rect_voltage),
                ("polar", :polar, ACPPowerModel, test_polar_voltage)]

function example_func(d, srating)
    return d + 20/srating*d^2
end

function sc_tests(filename)
    uc_filename = "$filename.pop_solution.json"
    model, sc_data_array, vars, lengths = ExaModelsPower.scopf_model(filename, uc_filename; backend=CUDABackend())
    @info "built model"
    result = madnlp(model; print_level = MadNLP.ERROR, tol=8e-3, linear_solver=MadNLPGPU.CUDSSSolver)
    JLD2.save("result.jld2", "solution", result, "vars", vars, "lens", lens)
    ExaModelsPower.save_go3_solution(uc_filename, "solution_go3", result, vars, lengths)
end

PowerModels.silence()

function parse_pm(filename)
    data = PowerModels.parse_file(filename)
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)

    return data
end

function runtests()
    @testset "ExaModelsPower test" begin

        for (T, backend) in CONFIGS
            for (filename, case, test_function) in test_cases
                data_pm = parse_pm(filename)
                for (form_str, form, power_model, test_voltage) in static_forms
                    m32, v32, c32 = opf_model(filename; T=Float32, backend = backend, form=form)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    va32, vm32, pg32, qg32, p32, q32 = v32

                    m64, v64, c64 = opf_model(filename; T=Float64, backend = backend, form=form)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    va64, vm64, pg64, qg64, p64, q64 = v64
                    
                    nlp_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>Float64(result64.options.tol), "print_level"=>0)
                    result_pm = solve_opf(filename, power_model, nlp_solver)

                    m_pm = JuMP.Model()
                    pm = instantiate_model(data_pm, power_model, PowerModels.build_opf, jump_model = m_pm)
                    nlp_pm = MathOptNLPModel(m_pm)
                    result_nlp_pm = madnlp(nlp_pm; print_level = MadNLP.ERROR)

                    @info form_str
                    @testset "$case, static, $backend, $form_str" begin
                        test_float32(m32, m64, result64, backend)
                        eval(test_function)(result64, result_pm, result_nlp_pm, pg64, qg64, p64, q64)
                        test_voltage(result64, result_pm, va64, vm64)
                    end
                end
            end
            
            #Test MP
            for (form_str, symbol) in [("rect", :rect), ("polar", :polar)]
                for (filename, case, Pd_pregen, Qd_pregen, true_sol_curve, true_sol_pregen) in mp_test_cases
                    #Curve = [1, .9, .8, .95, 1]

                    m32, v32, c32 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1]; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1]; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "$(case), MP, $(T), $(backend), curve, $(form_str)" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_curve)
                    end
                    #w function
                    m32, v32, c32 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1], example_func; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1], example_func; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "$(case), MP, $(T), $(backend), curve, $(form_str), func" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_curve)
                    end
                

                    #Pregenerated Pd and Qd
                    m32, v32, c32 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "$(case), MP, $(T), $(backend), pregen, $(form_str)" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_pregen)
                    end
                    #w function
                    m32, v32, c32 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen, example_func; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen, example_func; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "$(case), MP, $(T), $(backend), pregen, $(form_str), func" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_pregen)
                    end
                end
                
                # Test MP w storage
                for (filename, case, Pd_pregen, Qd_pregen, true_sol_curve_stor, 
                    true_sol_curve_stor_func, true_sol_pregen_stor, true_sol_pregen_stor_func) in mp_stor_test_cases
                    
                    m32, v32, c32 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1]; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1]; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "MP w storage, $(case), $(T), $(backend), curve, $(form_str)" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_curve_stor)
                    end

                    #With function
                    m32, v32, c32 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1], example_func; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, [1, .9, .8, .95, 1], example_func; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "MP w storage, $(case), $(T), $(backend), curve, $(form_str), func" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_curve_stor_func)
                    end

                    #Pregenerated Pd and Qd
                    m32, v32, c32 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "MP w storage, $(case), $(T), $(backend), pregen, $(form_str)" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_pregen_stor)
                    end

                    #With function
                    m32, v32, c32 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen, example_func; T = Float32, backend = backend, form = symbol)
                    result32 = madnlp(m32; print_level = MadNLP.ERROR)
                    m64, v64, c64 = eval(mpopf_model)(filename, Pd_pregen, Qd_pregen, example_func; T = Float64, backend = backend, form = symbol)
                    result64 = madnlp(m64; print_level = MadNLP.ERROR)
                    @testset "MP w storage, $(case), $(T), $(backend), pregen, $(form_str), func" begin
                        test_float32(m32, m64, result64, backend)
                        test_mp_case(result64, true_sol_pregen_stor_func)
                    end
                end
            end
        end
    end
end

runtests()
