function build_dcopf(data, user_callback; backend = nothing, T = Float64, core = nothing, kwargs...)

    core = isnothing(core) ? ExaCore(T; backend = backend, concrete = Val(true)) : core
    T, backend = typeof(core).parameters[1], core.backend

    @add_var(core, va, length(data.bus))

    @add_var(core, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)

    @add_var(core, pf, length(data.branch); lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, o, gen_cost(g, pg[g.i]) for g in data.gen)

    @add_con(core, c_ref_angle, c_ref_angle_polar(va[i]) for i in data.ref_buses)

    @add_con(core, c_ohms_law, c_ohms_law_dcopf(br, pf[br.i], va[br.f_bus], va[br.t_bus]) for br in data.branch)

    @add_con(core, c_phase_angle_diff, c_phase_angle_diff_polar(b, va[b.f_bus], va[b.t_bus]) for b in data.branch; lcon = data.angmin, ucon = data.angmax)

    @add_con(core, c_active_power_balance, c_active_power_balance_dc(b) for b in data.bus)
    @add_con!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(core, c_active_power_balance, br.f_bus => pf[br.i] for br in data.branch)
    @add_con!(core, c_active_power_balance, br.t_bus => -pf[br.i] for br in data.branch)

    vars = (
        va = va,
        pg = pg,
        pf = pf,
    )

    cons = (
        c_ref_angle = c_ref_angle,
        c_ohms_law = c_ohms_law,
        c_phase_angle_diff = c_phase_angle_diff,
        c_active_power_balance = c_active_power_balance,
    )


    core, vars2, cons2 = user_callback(core, vars, cons)
    model =ExaModel(core; kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)

    return model, vars, cons
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
    data = parse_ac_power_data(filename)
    data = convert_data(data, backend)

    return build_dcopf(data, user_callback; backend = backend, T = T, kwargs...)

end
