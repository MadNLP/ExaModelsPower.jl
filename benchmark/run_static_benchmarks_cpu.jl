using ExaModelsPower, MadNLP, MadNLPGPU, ExaModels
include("benchmark_opf.jl")
for coord in ["Polar", "Rectangular"]
    for tol in [1e-4, 1e-8]
        solve_static_cases(cases, tol, coord, "CPU";)
    end
end