convert_data(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(convert_array(d, backend) for d in data)

function parse_ac_power_data(filename)
    _, f = splitdir(filename)
    name, _ = splitext(f)

    @info "Loading matpower file"

    library = isfile(filename) ? nothing : :pglib
    data = ExaPowerIO.parse_matpower(filename; library)

    data = (
        baseMVA = [data.baseMVA],
        bus = data.bus,
        gen = data.gen,
        arc = data.arc,
        branch = data.branch,
        storage = isempty(data.storage) ? empty_data = Vector{NamedTuple{(:i,), Tuple{Int64}}}() : data.storage,
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
        pdmax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.charge_rating for s in data.storage],
        pcmax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.discharge_rating for s in data.storage],
        srating = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.thermal_rating for s in data.storage],
        emax = isempty(data.storage) ? Vector{NamedTuple{(:i,), Tuple{Int64}}}() : [s.energy_rating for s in data.storage],
    )

    return data
end
