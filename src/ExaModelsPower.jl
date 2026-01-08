module ExaModelsPower

import ExaModels: ExaModels, ExaCore, variable, constraint, ExaModel, objective, constraint!, convert_array, solution
using DelimitedFiles
using ExaPowerIO
using JSON

include("parser.jl")
include("constraint.jl")
include("opf.jl")
include("dcopf.jl")
include("goc3_parser.jl")
include("scopf.jl")
include("mpopf.jl")
include("sc_parser.jl")

const NAMES = filter(names(@__MODULE__; all = true)) do x
    str = string(x)
    endswith(str, "model") && !startswith(str, "#")
end

for name in filter(names(@__MODULE__; all = true)) do x
    endswith(string(x), "model")
end
    @eval export $name
end
    
function __init__()
    if haskey(ENV, "EXA_MODELS_DEPOT")
        global TMPDIR = ENV["EXA_MODELS_DEPOT"]
    else
        global TMPDIR = joinpath(@__DIR__,"..","data")
        mkpath(TMPDIR)
    end
end

end # module ExaModelsPower
