module ExaModelsPower

import ExaModels: ExaModels, ExaCore, variable, parameter, constraint, ExaModel, objective, constraint!, convert_array, solution
using DelimitedFiles
using ExaPowerIO
using JSON

# The builders and the public user_callback contract rely on the mutating API
# (variable/constraint/constraint!), which only exists for LegacyExaCore.
# ExaCore(T; backend) still returns one but emits a deprecation warning, so
# construct the wrapper directly until the package migrates to the functional
# add_var/add_con API. The element type comes from x0 rather than T because
# some backends promote it (e.g. Metal: Float64 -> Float32).
function legacy_core(::Type{T}, backend) where {T}
    inner = ExaCore(T; backend = backend, concrete = Val(true))
    return ExaModels.LegacyExaCore{eltype(inner.x0),typeof(inner.x0),typeof(backend),typeof(inner.tag)}(inner)
end

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
