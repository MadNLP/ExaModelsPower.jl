using ExaModelsPower, MadNLP, MadNLPGPU, ExaModels
include("benchmark_opf.jl")
for tol in [1e-4, 1e-8]
    solve_benchmark_cases(cases, tol, "CPU"; mp = true)
end