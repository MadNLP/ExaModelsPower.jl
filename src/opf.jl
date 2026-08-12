function dummy_extension(core, vars, cons)
    return core, (;), (;)
end

# ── Recipe, arguments, model ──────────────────────────────────────────────────
#
# The AC OPF is written once, as a *recipe*: an `ExaCore` whose structure is
# fixed but whose data is left open, standing in as an `ExaModels.ArgSource`
# placeholder.  `ac_opf_args` produces the data that closes it, and
# `ac_opf_model` is the two composed — so there is one definition of the model
# rather than two that can drift.
#
# The reason for the split is ahead-of-time compilation.  Writing a model
# against placeholders *is* what `ExaModelsC` needs in order to compile it into
# a shared library with the data left open; the author never has to reason
# about what `juliac --trim=safe` can digest.
#
# ── What the placeholder cannot do, the caller supplies ──
#
# A placeholder supports field access and deferred arithmetic — `data.bus`,
# `length(data.gen)`, `-data.rate_a` — and nothing else.  An array built to a
# deferred size, a broadcast, or a comprehension has no symbolic form, and
# `ExaModels` refuses them rather than silently mis-building.  Three things in
# this model were written that way before, and none of them is dropped: each
# becomes a value `ac_opf_args` computes and passes in as ordinary data.
#
#   * starting points, previously `fill!(similar(data.bus, T), one(T))`;
#   * the rectangular form's squared voltage bounds, previously `data.vmin.^2`;
#   * the thermal limits' `-Inf` lower bounds, previously a filled array.
#
# `ac_opf_args` is the adapter between what a *user* has — a case file, and
# preferences about how to start — and what the recipe (and, past it, the
# ExaModelsC boundary) can carry, which is data and only data.

"""
    opf_start(value, n, T)

Materialize a user-supplied starting point into a length-`n` `Vector{T}`.

`value` may be
- a scalar — the flat start, e.g. `one(T)` for voltage magnitude;
- an `AbstractVector` of length `n`;
- a `Base.Generator`, e.g. `Base.Generator(f, 1:n)` — note that `f` should be a
  *named* function rather than an anonymous one, so that the type survives
  ahead-of-time compilation and reads intelligibly in errors.

Generators and vectors are collected here, in `ac_opf_args`, rather than being
referred to from the recipe: only data crosses into an instantiated model, so
resolving them at this point is what lets the same start work for an in-Julia
model and for a compiled library.
"""
opf_start(v::Number, n, ::Type{T}) where {T} = fill(T(v), n)
opf_start(g::Base.Generator, n, ::Type{T}) where {T} = _opf_start_check(collect(T, g), n)
opf_start(v::AbstractVector, n, ::Type{T}) where {T} =
    _opf_start_check(convert(Vector{T}, collect(v)), n)

function _opf_start_check(v, n)
    length(v) == n || throw(
        DimensionMismatch(
            "a starting point of length $(length(v)) was given for a variable " *
            "block of length $n",
        ),
    )
    return v
end

# The defaults are the ones this model has always used: a flat start of 1.0 for
# voltage magnitude, zero everywhere else (`zero(T)` is also `ExaModels`' own
# default for a variable with no `start`).
_default_start(::Type{T}) where {T} =
    (va = zero(T), vm = one(T), vr = one(T), vim = zero(T),
     pg = zero(T), qg = zero(T), p = zero(T), q = zero(T))

# A start entry may also be given as a function of the parsed data, which is how
# a caller warm-starts from the case's own operating point — `start = (; vm = d
# -> d.vm0, va = d -> d.va0)`.  Resolved before materialization, so the result
# is still ordinary data.
_resolve_start(f, data) = f(data)
_resolve_start(v::Union{Number,AbstractVector,Base.Generator}, data) = v

"""
    ac_opf_args(filename; T = Float64, backend = nothing, start = (;))
        -> (data,)

Return the argument tuple that closes [`ac_opf_recipe`](@ref) — the parsed case
together with everything the recipe cannot compute from a placeholder.

`start` overrides any of the starting points `va`, `vm`, `vr`, `vim`, `pg`,
`qg`, `p`, `q`. Each entry may be a scalar, a vector, a `Base.Generator`, or a
function of the parsed data (`d -> d.vm0` to warm-start from the case's own
operating point). Defaults are a flat `1.0` voltage magnitude and zero
elsewhere, which is what this model used before the recipe split.

# Example
```julia
args = ac_opf_args("pglib_opf_case118_ieee.m"; start = (vm = d -> d.vm0, va = d -> d.va0))
model = ExaModel(ac_opf_recipe()[1], args...)
```
"""
ac_opf_args(filename; T = Float64, backend = nothing, start = (;)) =
    ac_opf_args(filename, T, backend, start)

# The positional method is the one a compiled library calls: `::Type{T}` makes
# `T` a static parameter, where the keyword form leaves it a `Type`-typed value
# that nothing downstream can specialize on.
function ac_opf_args(filename, ::Type{T}, backend = nothing, start = (;)) where {T}
    p = parse_ac_power_data(filename, T)
    s = merge(_default_start(T), NamedTuple(start))
    bus, gen, arc, branch = p.bus, p.gen, p.arc, p.branch
    nbus, ngen, narc, nbranch = length(bus), length(gen), length(arc), length(branch)

    # EXACTLY the fields the model bodies read, and no more. `map` over a
    # NamedTuple — which is what `convert_data` is — stays inferable up to 31
    # fields and gives up at 32 (measured, this Julia); passing the parse
    # through whole came to 35, and the result was a `NamedTuple` with concrete
    # names and no value types at all. That one abstract value was enough to
    # leave `ExaModel(CORE, data)` unresolved under `--trim=safe`.
    #
    # Nothing is lost by narrowing: the fields dropped here — `baseMVA`,
    # `storage`, `vm0`/`va0`/`pg0`/`qg0` and the storage ratings — are inputs to
    # the starting points, which are resolved just below and travel as the
    # `*_start` arrays.
    data = (;
        bus, gen, arc, branch,
        ref_buses = p.ref_buses,
        vmin = p.vmin, vmax = p.vmax,
        pmin = p.pmin, pmax = p.pmax,
        qmin = p.qmin, qmax = p.qmax,
        angmin = p.angmin, angmax = p.angmax,
        rate_a = p.rate_a,
        # Broadcasting a placeholder is refused by design, so the rectangular
        # form's squared voltage bounds are computed here, beside the parse.
        vmin2 = p.vmin .^ 2,
        vmax2 = p.vmax .^ 2,
        # The thermal limits are one-sided; the bound array used to be built
        # with `fill!(similar(...), -Inf)` against a length only known later.
        branch_ninf = fill(T(-Inf), nbranch),
        # Resolved against the FULL parse, so `start = (vm = d -> d.vm0,)`
        # still reaches a field the model itself never sees.
        va_start = opf_start(_resolve_start(s.va, p), nbus, T),
        vm_start = opf_start(_resolve_start(s.vm, p), nbus, T),
        vr_start = opf_start(_resolve_start(s.vr, p), nbus, T),
        vim_start = opf_start(_resolve_start(s.vim, p), nbus, T),
        pg_start = opf_start(_resolve_start(s.pg, p), ngen, T),
        qg_start = opf_start(_resolve_start(s.qg, p), ngen, T),
        p_start = opf_start(_resolve_start(s.p, p), narc, T),
        q_start = opf_start(_resolve_start(s.q, p), narc, T),
    )

    return (convert_data(data, backend),)
end

# ── The model bodies ─────────────────────────────────────────────────────────
#
# Each takes the core and the data and returns the core back.  Handing the core
# back is not optional: `@add_var` and `@add_con` *rebind* their first argument,
# so a caller that keeps its own binding is left with a core carrying the blocks
# but stale counters — which fails much later, at instantiation, with
# `Nonsensical dimensions` and nothing pointing at the cause.
#
# `data` here is either an `ArgSource` placeholder (the recipe) or a concrete
# `NamedTuple` (the fixed core).  The body is identical for both, which is what
# keeps the compiled and in-Julia forms from drifting.

function build_polar_opf(core, data, user_callback, ::Type{T}) where {T}
    @add_var(core, va, length(data.bus); start = data.va_start)
    @add_var(core, vm,
        length(data.bus);
        start = data.vm_start,
        lvar = data.vmin,
        uvar = data.vmax,
    )

    @add_var(core, pg, length(data.gen); start = data.pg_start, lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg, length(data.gen); start = data.qg_start, lvar = data.qmin, uvar = data.qmax)

    @add_var(core, p, length(data.arc); start = data.p_start, lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q, length(data.arc); start = data.q_start, lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, o, gen_cost(g, pg[g.i]) for g in data.gen)

    @add_con(core, c_ref_angle, c_ref_angle_polar(va[i]) for i in data.ref_buses)

    @add_con(core, c_to_active_power_flow, c_to_active_power_flow_polar(b, p[b.f_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    @add_con(core, c_to_reactive_power_flow, c_to_reactive_power_flow_polar(b, q[b.f_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    @add_con(core, c_from_active_power_flow, c_from_active_power_flow_polar(b, p[b.t_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    @add_con(core, c_from_reactive_power_flow, c_from_reactive_power_flow_polar(b, q[b.t_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    @add_con(core, c_phase_angle_diff,
        c_phase_angle_diff_polar(b,va[b.f_bus],va[b.t_bus]) for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    @add_con(core, c_active_power_balance, c_active_power_balance_demand_polar(b, vm[b.i]) for b in data.bus)

    @add_con(core, c_reactive_power_balance, c_reactive_power_balance_demand_polar(b, vm[b.i]) for b in data.bus)

    @add_con!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    @add_con!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    @add_con!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    @add_con(core, c_from_thermal_limit, c_thermal_limit(b,p[b.f_idx],q[b.f_idx]) for b in data.branch;
        lcon = data.branch_ninf,
        )

    @add_con(core, c_to_thermal_limit, c_thermal_limit(b,p[b.t_idx],q[b.t_idx])
        for b in data.branch;
        lcon = data.branch_ninf,
    )

    vars = (
            va = va,
            vm = vm,
            pg = pg,
            qg = qg,
            p = p,
            q = q
        )

    cons = (
        c_ref_angle = c_ref_angle,
        c_to_active_power_flow = c_to_active_power_flow,
        c_to_reactive_power_flow = c_to_reactive_power_flow,
        c_from_active_power_flow = c_from_active_power_flow,
        c_from_reactive_power_flow = c_from_reactive_power_flow,
        c_phase_angle_diff = c_phase_angle_diff,
        c_active_power_balance = c_active_power_balance,
        c_reactive_power_balance = c_reactive_power_balance,
        c_from_thermal_limit = c_from_thermal_limit,
        c_to_thermal_limit = c_to_thermal_limit
    )

    core, vars2, cons2 = user_callback(core, vars, cons)

    return core, (;vars..., vars2...), (;cons..., cons2...)
end

function build_rect_opf(core, data, user_callback, ::Type{T}) where {T}
    @add_var(core, vr, length(data.bus); start = data.vr_start)
    @add_var(core, vim, length(data.bus); start = data.vim_start)

    @add_var(core, pg, length(data.gen); start = data.pg_start, lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg, length(data.gen); start = data.qg_start, lvar = data.qmin, uvar = data.qmax)

    @add_var(core, p, length(data.arc); start = data.p_start, lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q, length(data.arc); start = data.q_start, lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, o, gen_cost(g, pg[g.i]) for g in data.gen)

    @add_con(core, c_ref_angle, c_ref_angle_rect(vr[i], vim[i]) for i in data.ref_buses)

    @add_con(core, c_to_active_power_flow, c_to_active_power_flow_rect(b,p[b.f_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    @add_con(core, c_to_reactive_power_flow, c_to_reactive_power_flow_rect(b,q[b.f_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    @add_con(core, c_from_active_power_flow, c_from_active_power_flow_rect(b,p[b.t_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    @add_con(core, c_from_reactive_power_flow, c_from_reactive_power_flow_rect(b,q[b.t_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    @add_con(core, c_phase_angle_diff, c_phase_angle_diff_rect(b,
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus])
        for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    @add_con(core, c_active_power_balance, c_active_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus)

    @add_con(core, c_reactive_power_balance, c_reactive_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus)

    @add_con!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    @add_con!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    @add_con!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    @add_con(core, c_from_thermal_limit, c_thermal_limit(b,p[b.f_idx], q[b.f_idx]) for b in data.branch;
        lcon = data.branch_ninf,
    )

    @add_con(core, c_to_thermal_limit, c_thermal_limit(b,p[b.t_idx], q[b.t_idx])
        for b in data.branch;
        lcon = data.branch_ninf,
    )

    @add_con(core, c_voltage_magnitude, c_voltage_magnitude_rect(vr[b.i], vim[b.i]) for b in data.bus;
        lcon = data.vmin2,
        ucon = data.vmax2
    )

    vars = (
        vr = vr,
        vim = vim,
        pg = pg,
        qg = qg,
        p = p,
        q = q
    )

    cons = (
        c_ref_angle = c_ref_angle,
        c_to_active_power_flow = c_to_active_power_flow,
        c_to_reactive_power_flow = c_to_reactive_power_flow,
        c_from_active_power_flow = c_from_active_power_flow,
        c_from_reactive_power_flow = c_from_reactive_power_flow,
        c_phase_angle_diff = c_phase_angle_diff,
        c_active_power_balance = c_active_power_balance,
        c_reactive_power_balance = c_reactive_power_balance,
        c_from_thermal_limit = c_from_thermal_limit,
        c_to_thermal_limit = c_to_thermal_limit,
        c_voltage_magnitude = c_voltage_magnitude
    )

    core, vars2, cons2 = user_callback(core, vars, cons)

    return core, (;vars..., vars2...), (;cons..., cons2...)
end

_opf_body(form) =
    form == :polar ? build_polar_opf :
    form == :rect ? build_rect_opf :
    error("Invalid coordinate symbol - valid options are :polar or :rect")

"""
    ac_opf_recipe(; backend, T, form, user_callback) -> (core, variables, constraints)

Return the AC OPF *recipe* — an `ExaCore` holding the model's structure with its
data left open — together with the variable and constraint handles.

Close it with [`ac_opf_args`](@ref):

```julia
core, vars, cons = ac_opf_recipe()
model = ExaModel(core, ac_opf_args("pglib_opf_case118_ieee.m")...)
```

The same recipe instantiates at any case; it is not consumed by the first use.
See [`ac_opf_core`](@ref) for the form `ExaModelsC` compiles.

# Arguments
- `backend`: the array backend to build against. Default `nothing` (CPU).
- `T`: the numeric type (default `Float64`).
- `form`: voltage representation, `:polar` or `:rect`. Default `:polar`.
- `user_callback`: user function that extends the model.
"""
function ac_opf_recipe(;
    backend = nothing,
    T = Float64,
    form = :polar,
    user_callback = dummy_extension,
)
    core, data = ExaCore(T; backend = backend, nargs = Val(1))
    return _opf_body(form)(core, data, user_callback, T)
end

"""
    ac_opf_core(filename; backend, T, form, user_callback, start)
        -> (core, variables, constraints)

Return an AC OPF core with the case's data already in it — the same model as
[`ac_opf_recipe`](@ref) closed by [`ac_opf_args`](@ref), but built eagerly, so
the core declares no placeholders (`nargs = Val(0)`).

This is the form `ExaModelsC.compile_library` compiles: one library per case,
with no instantiation data crossing the C boundary.

```julia
core, _, _ = ac_opf_core("pglib_opf_case118_ieee.m")
compile_library("@acopf118", core)      # note: no example argument
```

Passing an example argument there would select the recipe path instead and be
refused for carrying tables rather than a single integer — a different failure
that reads like this one.
"""
function ac_opf_core(
    filename;
    backend = nothing,
    T = Float64,
    form = :polar,
    user_callback = dummy_extension,
    start = (;),
)
    data, = ac_opf_args(filename; T = T, backend = backend, start = start)
    core = ExaCore(T; backend = backend)
    return _opf_body(form)(core, data, user_callback, T)
end

"""
    ac_opf_model(filename; backend, T, form, user_callback, start, kwargs...)

Return `ExaModel`, variables, and constraints for a static AC Optimal Power Flow
(ACOPF) problem from the given file.

Defined as [`ac_opf_recipe`](@ref) instantiated at [`ac_opf_args`](@ref), so the
model solved here and the model compiled by `ExaModelsC` are built from one
definition rather than two.

# Arguments
- `filename::String`: Path to the data file.
- `backend`: The solver backend to use. Default if nothing.
- `T`: The numeric type to use (default is `Float64`).
- `form`: Voltage representation, either `:polar` or `:rect`. Default is `:polar`.
- `user_callback`: User function that extends the model
- `start`: Starting-point overrides; see [`ac_opf_args`](@ref).
- `kwargs...`: Additional keyword arguments passed to the model builder.

# Returns
A vector `(model, variables, constraints)`:
- `model`: An `ExaModel` object.
- `variables`: NamedTuple of model variables.
- `constraints`: NamedTuple of model constraints.
"""
function ac_opf_model(
    filename;
    backend = nothing,
    T = Float64,
    form = :polar,
    user_callback = dummy_extension,
    start = (;),
    kwargs...,
)
    core, vars, cons =
        ac_opf_recipe(; backend = backend, T = T, form = form, user_callback = user_callback)
    args = ac_opf_args(filename; T = T, backend = backend, start = start)
    return ExaModel(core, args...; prod = true, kwargs...), vars, cons
end
