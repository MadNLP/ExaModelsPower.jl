convert_data(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(convert_array(d, backend) for d in data)

function _resolve_power_data_path(filename)
    isfile(filename) && return filename
    return joinpath(ExaPowerIO.get_path(:pglib), filename)
end

function _parser_symbol(parser)
    parser isa Symbol && return parser
    parser isa AbstractString && return Symbol(parser)
    error("parser must be :exapowerio or :powerio")
end

function _empty_storage_data()
    return Vector{NamedTuple{(:i,), Tuple{Int64}}}()
end

function _shape_exapowerio_data(data)
    empty_storage = _empty_storage_data()
    return (
        baseMVA = [data.baseMVA],
        bus = data.bus,
        gen = data.gen,
        arc = data.arc,
        branch = data.branch,
        storage = isempty(data.storage) ? empty_storage : data.storage,
        ref_buses = [i for i in 1:length(data.bus) if data.bus[i].type == 3],
        vmax = [bu.vmax for bu in data.bus],
        vmin = [bu.vmin for bu in data.bus],
        pmax = [g.pmax for g in data.gen],
        pmin = [g.pmin for g in data.gen],
        qmax = [g.qmax for g in data.gen],
        qmin = [g.qmin for g in data.gen],
        angmax = [br.angmax for br in data.branch],
        angmin = [br.angmin for br in data.branch],
        rate_a = [a.rate_a for a in data.arc],
        vm0 = [b.vm for b in data.bus],
        va0 = [b.va for b in data.bus],
        pg0 = [g.pg for g in data.gen],
        qg0 = [g.qg for g in data.gen],
        pdmax = isempty(data.storage) ? empty_storage : [s.charge_rating for s in data.storage],
        pcmax = isempty(data.storage) ? empty_storage : [s.discharge_rating for s in data.storage],
        srating = isempty(data.storage) ? empty_storage : [s.thermal_rating for s in data.storage],
        emax = isempty(data.storage) ? empty_storage : [s.energy_rating for s in data.storage],
    )
end

function _parse_ac_power_data_exapowerio(filename, T; from)
    from === nothing || error("from requires parser = :powerio")
    @info "Loading matpower file"
    library = isfile(filename) ? nothing : :pglib
    return _shape_exapowerio_data(ExaPowerIO.parse_matpower(T, filename; library))
end

function _parse_ac_power_data_powerio(filename, T; from)
    path = _resolve_power_data_path(filename)
    @info "Loading power case file"
    return PowerIO.parse_ac_power_data(path; from = from, T = T)
end

function parse_ac_power_data(filename, T = Float64; from = nothing, parser = :powerio)
    parser = _parser_symbol(parser)
    if parser == :exapowerio
        return _parse_ac_power_data_exapowerio(filename, T; from = from)
    elseif parser == :powerio
        return _parse_ac_power_data_powerio(filename, T; from = from)
    else
        error("unknown parser $(repr(parser)); expected :exapowerio or :powerio")
    end
end
