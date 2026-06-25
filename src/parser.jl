convert_data(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(convert_array(d, backend) for d in data)

function _resolve_power_data_path(filename)
    isfile(filename) && return filename
    return joinpath(ExaPowerIO.get_path(:pglib), filename)
end

function parse_ac_power_data(filename, T = Float64; from = nothing)
    path = _resolve_power_data_path(filename)
    @info "Loading power case file"
    return PowerIO.parse_ac_power_data(path; from = from, T = T)
end
