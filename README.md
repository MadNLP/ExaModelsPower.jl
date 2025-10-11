# ExaModelsPower.jl
ExaModelsPower.jl is an optimal power flow models using ExaModels.jl

[![CI](https://github.com/MadNLP/ExaModelsPower.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MadNLP/ExaModelsPower.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://madsuite.org/ExaModelsPower.jl/stable/) 
[![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://madsuite.org/ExaModelsPower.jl/dev/) 
[![codecov](https://codecov.io/gh/MadNLP/ExaModelsPower.jl/graph/badge.svg?token=ybOObxcXhB)](https://codecov.io/gh/MadNLP/ExaModelsPower.jl)

## Usage
### Static optimal power flow
```julia
using ExaModelsPower, MadNLP, MadNLPGPU, CUDA, ExaModels, GOC3Benchmark, JSON


model, vars, cons = opf_model(
    "pglib_opf_case118_ieee.m";
    backend = CUDABackend(),
    form = :polar
)
result = madnlp(model; tol=1e-6)
```

### Security-constrained optimal power flow
```julia
#This model is based on the GOC3 formulation of the SCOPF problem
#https://www.pnnl.gov/publications/grid-optimization-competition-challenge-3-problem-formulation

#The current implementation requires a UC solution to be provided, which is then parsed with
#the other input data to generate a structure of named tuples which can then interface with 
#ExaModels to generate the full model. We do not make any relaxations or decompositions for this problem

model, sc_data, vars, lengths = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-4)

#Solution from GPU can be used to warm start a CPU solution or vice versa
model_cpu, sc_data, vars, lengths = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    result_set = [result, vars]
)
result_cpu = ipopt(model_cpu; tol=1e-8)

#Additionally, the SC problem can be evaluated without contingencies
model, sc_data, vars, lengths = goc3_model(
    "data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"; 
    backend = CUDABackend(), include_ctg = false
)
result = madnlp(model; tol=1e-4)
```

### Multi-period optimal power flow
```julia
model, vars, cons = mpopf_model(
    "pglib_opf_case118_ieee.m", # static network data
    "/home/sshin/git/ExaModels_Multiperiod/data/case118_onehour_168.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/case118_onehour_168.Qd"; # dynamic load data
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)

#Alternatively, input a vector to scale baseline demand to generate a demand curve
model, vars, cons = mpopf_model(
    "pglib_opf_case118_ieee.m", # static network data
    [.64, .60, .58, .56, .56, .58, .64, .76, .87, .95, .99, 1.0, .99, 1.0, 1.0,
    .97, .96, .96, .93, .92, .92, .93, .87, .72, .64], #Demand curve
    backend = CUDABackend(),
    corrective_action_ratio = 0.3
)
result = madnlp(model; tol=1e-6)

#mpopf_model can also handle inputs with storage constraints
model, vars, cons = mpopf_model(
    "pglib_opf_case30_ieee_mod.m", # static network data with storage parameters
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Qd"; # dynamic load data
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)

#Alternatively, provide a smooth function for the charge/discharge efficiency to remove complementarity constraint
function example_func(d, srating)
    return -((s_rating/2)^d)+1
end

model, vars, cons = mpopf_model(
    "pglib_opf_case30_ieee_mod.m", # static network data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Pd", # dynamic load data
    "/home/sshin/git/ExaModels_Multiperiod/data/halfhour_30.Qd"; # dynamic load data
    example_func, #Discharge/charge efficiency modeled along smooth curve
    backend = CUDABackend()
)
result = madnlp(model; tol=1e-6)


#Modified datasets that can be used for testing
#https://github.com/mit-shin-group/multi-period-opf-data
```




