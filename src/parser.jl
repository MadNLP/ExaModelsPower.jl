# `map`, not `NamedTuple{names}(generator)`. The generator form builds the
# tuple from an iterator whose element types inference cannot pin one by one,
# so the result is a `NamedTuple{names}` with no value types — concrete names,
# abstract contents. `map` over a NamedTuple is inferable elementwise.
# Recurses into a nested NamedTuple. The multi-period arguments group their
# N-expanded bound matrices under one field, because there are enough of them
# to push a flat tuple past the 31-field limit above.
convert_data(data::NamedTuple, backend) = map(d -> _convert_field(d, backend), data)
_convert_field(d, backend) = convert_array(d, backend)
_convert_field(d::NamedTuple, backend) = convert_data(d, backend)

# `T` is a TYPE PARAMETER, not a value of type `Type`. As a plain positional
# argument it arrives typed `Type` — abstract — and then `parse_matpower(T, …)`
# cannot specialize, so its result is an unparameterized `PowerData` and every
# field read off it widens to `AbstractArray{BusData{T}} where T`. Written
# `::Type{T}` it is a static parameter and the whole chain stays concrete.
parse_ac_power_data(filename) = parse_ac_power_data(filename, Float64)
function parse_ac_power_data(filename, ::Type{T}) where {T}
    _, f = splitdir(filename)
    name, _ = splitext(f)

    @info "Loading matpower file"

    # Branch on the CALL, not on the keyword. A `library` of
    # `Union{Nothing,Symbol}` leaves `parse_matpower`'s return type
    # uninferable, and every field assembled from it below then infers as
    # `Any`. Measured: given either value concretely, `parse_matpower` returns
    # a fully concrete `PowerData{Float64, Vector{BusData{Float64}}, …}` — so
    # the instability was ours, not the parser's.
    raw =
        isfile(filename) ? ExaPowerIO.parse_matpower(T, filename; library = nothing) :
        ExaPowerIO.parse_matpower(T, filename; library = :pglib)

    # `raw`, not a second binding called `data`. Assigning the same name twice
    # while a closure captures it — the `ref_buses` comprehension does — makes
    # Julia box it as a `Core.Box`, whose contents are `Any`. That single box
    # was what made the whole function infer as `Any`, and it is invisible in
    # the source: both assignments are perfectly ordinary.
    #
    # The storage fields below are NOT special-cased on emptiness. A ternary
    # whose branches return different types (`Vector{NamedTuple{(:i,)}}` when
    # empty, `Vector{T}` otherwise) makes this whole NamedTuple non-concrete,
    # which is the other half of the same instability. An empty
    # `Vector{StorageData{T}}` iterates to an empty `Vector{T}` on its own, so
    # the guard bought nothing and cost the return type.
    data = (
        baseMVA = [raw.baseMVA],
        bus = raw.bus,
        gen = raw.gen,
        arc = raw.arc,
        branch = raw.branch,
        storage = raw.storage,
        ref_buses = [i for i in 1:length(raw.bus) if raw.bus[i].type == 3],
        vmax = [bu.vmax for bu in raw.bus],
        vmin = [bu.vmin for bu in raw.bus],
        pmax = [g.pmax for g in raw.gen],
        pmin = [g.pmin for g in raw.gen],
        qmax = [g.qmax for g in raw.gen],
        qmin = [g.qmin for g in raw.gen],
        angmax = [br.angmax for br in raw.branch],
        angmin = [br.angmin for br in raw.branch],
        rate_a = [a.rate_a for a in raw.arc],
        vm0 = [b.vm for b in raw.bus],
        va0 = [b.va for b in raw.bus],
        pg0 = [g.pg for g in raw.gen],
        qg0 = [g.qg for g in raw.gen],
        pdmax = [s.charge_rating for s in raw.storage],
        pcmax = [s.discharge_rating for s in raw.storage],
        srating = [s.thermal_rating for s in raw.storage],
        emax = [s.energy_rating for s in raw.storage],
    )

    return data
end
