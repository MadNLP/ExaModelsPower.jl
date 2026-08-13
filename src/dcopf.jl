# Takes the core and hands it back: `@add_var` and `@add_con` rebind their
# first argument, so a caller keeping its own binding is left with the blocks
# and stale counters. `data` is either an `ArgSource` placeholder (the recipe)
# or a concrete NamedTuple (the eager core) — one body serves both.
# ── DC: the methods that differ from the AC spine ───────────────────────────
#
# The body is `build_opf` in opf.jl; only these differ. DC has no reactive
# half, its flows are per branch rather than per arc, and it has no thermal
# limits — so `add_extras!` is where the merge would start costing more than it
# saves if a formulation needed several such no-ops. It needs one.

function add_voltage!(core, ::DC, d)
    @add_var(core, va, length(d.bus); start = d.va_start)
    return core, (; va)
end

function add_generation!(core, ::DC, d)
    @add_var(core, pg, length(d.gen); start = d.pg_start, lvar = d.pmin, uvar = d.pmax)
    return core, (; pg)
end

function add_flows!(core, ::DC, d)
    @add_var(core, pf, length(d.branch); start = d.pf_start,
        lvar = -d.branch_rate_a, uvar = d.branch_rate_a)
    return core, (; pf)
end

@inline c_ref(::DC, V, i) = c_ref_angle_polar(V.va[i])
@inline c_angle(::DC, b, V) = c_phase_angle_diff_polar(b, V.va[b.f_bus], V.va[b.t_bus])
# The DC active balance takes no voltage; `V` is carried for uniformity with
# the other formulations.
@inline c_bal_p(::DC, b, V) = c_active_power_balance_dc(b)

function add_flow_constraints!(core, ::DC, d, V, F)
    @add_con(core, c_ohms_law,
        c_ohms_law_dcopf(br, F.pf[br.i], V.va[br.f_bus], V.va[br.t_bus]) for br in d.branch)
    return core, (; c_ohms_law)
end

function add_balance!(core, form::DC, d, V, G, F)
    @add_con(core, c_active_power_balance, c_bal_p(form, b, V) for b in d.bus)
    @add_con!(core, c_active_power_balance, g.bus => -G.pg[g.i] for g in d.gen)
    @add_con!(core, c_active_power_balance, br.f_bus => F.pf[br.i] for br in d.branch)
    @add_con!(core, c_active_power_balance, br.t_bus => -F.pf[br.i] for br in d.branch)
    return core, (; c_active_power_balance)
end

add_extras!(core, ::DC, d, V, F) = (core, (;))

"""
    dcopf_recipe(; backend, T, user_callback) -> (core, variables, constraints)

The DC OPF *recipe* — structure with the case data left open. Close it with
[`ac_opf_args`](@ref), which assembles what a placeholder cannot compute:

```julia
core, vars, cons = dcopf_recipe()
model = ExaModel(core, ac_opf_args("pglib_opf_case118_ieee.m")...)
```
"""
function dcopf_recipe(; backend = nothing, T = Float64, user_callback = dummy_extension)
    core, data = ExaCore(T; backend = backend, nargs = Val(1))
    return build_opf(core, DC(), data, user_callback, T)
end

"""
    dcopf_core(filename; backend, T, user_callback, start)
        -> (core, variables, constraints)

The same DC model built eagerly, so the core declares no placeholders — the
form `ExaModelsC.compile_library` compiles as a fixed model.
"""
function dcopf_core(filename; backend = nothing, T = Float64,
                    user_callback = dummy_extension, start = (;))
    data, = ac_opf_args(filename; T = T, backend = backend, start = start)
    core = ExaCore(T; backend = backend)
    return build_opf(core, DC(), data, user_callback, T)
end


"""
    dcopf_model(filename; backend, T, user_callback)

Return `ExaModel`, variables, and constraints for a static linearized DC Optimal Power Flow (DCOPF) problem from the given file.

# Arguments
- `filename::String`: Path to the data file.
- `backend`: The solver backend to use. Default if nothing.
- `T`: The numeric type to use (default is `Float64`).
- `user_callback`: User function that extends the model
- `kwargs...`: Additional keyword arguments passed to the model builder.

# Returns
A vector `(model, variables, constraints)`:
- `model`: An `ExaModel` object.
- `variables`: NamedTuple of model variables.
- `constraints`: NamedTuple of model constraints.
"""

function dcopf_model(
    filename;
    backend = nothing,
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)
    core, vars, cons = dcopf_recipe(; backend = backend, T = T, user_callback = user_callback)
    args = ac_opf_args(filename; T = T, backend = backend)
    model = ExaModel(core, args...; prod = true, kwargs...)
    # Resolved against the same arguments the model was built from; a recipe's
    # handles carry `ArgNode` offsets that `solution` cannot index with.
    return model,
           ExaModels.instantiate(vars, args...),
           ExaModels.instantiate(cons, args...)

end
