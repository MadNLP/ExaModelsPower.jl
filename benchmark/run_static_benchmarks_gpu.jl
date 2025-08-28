using ExaModelsPower, MadNLP, MadNLPGPU, CUDA, ExaModels
println("Using device: ", CUDA.device())
include("benchmark_opf.jl")
for coord in ["Polar", "Rectangular"]
    for tol in [1e-4, 1e-8]
            solve_benchmark_cases(cases, tol, "GPU"; coords = coord)
        end
    end
end