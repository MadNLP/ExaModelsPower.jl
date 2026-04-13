function dummy_extension(core, vars, cons)
    return (;), (;)
end

function build_polar_opf(data, user_callback; backend = nothing, T=Float64, kwargs...)
    core = ExaCore(T; backend = backend, concrete = Val(true))

    @add_var(core, va, length(data.bus))

    @add_var(
        core,
        vm,
        length(data.bus);
        start = fill!(similar(data.bus, Float64), 1.0),
        lvar = data.vmin,
        uvar = data.vmax,
    )

    @add_var(core, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    @add_var(core, p, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, o, gen_cost(g, pg[g.i]) for g in data.gen)

    @add_con(core, c_ref_angle, c_ref_angle_polar(va[i]) for i in data.ref_buses)

    @add_con(
        core,
        c_to_active_power_flow,
        c_to_active_power_flow_polar(
            b,
            p[b.f_idx],
            vm[b.f_bus],
            vm[b.t_bus],
            va[b.f_bus],
            va[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_to_reactive_power_flow,
        c_to_reactive_power_flow_polar(
            b,
            q[b.f_idx],
            vm[b.f_bus],
            vm[b.t_bus],
            va[b.f_bus],
            va[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_from_active_power_flow,
        c_from_active_power_flow_polar(
            b,
            p[b.t_idx],
            vm[b.f_bus],
            vm[b.t_bus],
            va[b.f_bus],
            va[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_from_reactive_power_flow,
        c_from_reactive_power_flow_polar(
            b,
            q[b.t_idx],
            vm[b.f_bus],
            vm[b.t_bus],
            va[b.f_bus],
            va[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_phase_angle_diff,
        c_phase_angle_diff_polar(b, va[b.f_bus], va[b.t_bus]) for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    @add_con(
        core,
        c_active_power_balance,
        c_active_power_balance_demand_polar(b, vm[b.i]) for b in data.bus
    )

    @add_con(
        core,
        c_reactive_power_balance,
        c_reactive_power_balance_demand_polar(b, vm[b.i]) for b in data.bus
    )

    @add_con!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    @add_con!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    @add_con!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    @add_con(
        core,
        c_from_thermal_limit,
        c_thermal_limit(b, p[b.f_idx], q[b.f_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    @add_con(
        core,
        c_to_thermal_limit,
        c_thermal_limit(b, p[b.t_idx], q[b.t_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
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

    vars2, cons2 = user_callback(core, vars, cons)
    model =ExaModel(core; kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)

    return model, vars, cons
end

function build_rect_opf(data, user_callback; backend = nothing, T=Float64, kwargs...)
    core = ExaCore(T; backend = backend, concrete = Val(true))

    @add_var(
        core,
        vr,
        length(data.bus);
        start = fill!(similar(data.bus, Float64), 1.0),
    )

    @add_var(core, vim, length(data.bus))

    @add_var(core, pg, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    @add_var(core, p, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    @add_obj(core, o, gen_cost(g, pg[g.i]) for g in data.gen)

    @add_con(core, c_ref_angle, c_ref_angle_rect(vr[i], vim[i]) for i in data.ref_buses)

    @add_con(
        core,
        c_to_active_power_flow,
        c_to_active_power_flow_rect(
            b,
            p[b.f_idx],
            vr[b.f_bus],
            vr[b.t_bus],
            vim[b.f_bus],
            vim[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_to_reactive_power_flow,
        c_to_reactive_power_flow_rect(
            b,
            q[b.f_idx],
            vr[b.f_bus],
            vr[b.t_bus],
            vim[b.f_bus],
            vim[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_from_active_power_flow,
        c_from_active_power_flow_rect(
            b,
            p[b.t_idx],
            vr[b.f_bus],
            vr[b.t_bus],
            vim[b.f_bus],
            vim[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_from_reactive_power_flow,
        c_from_reactive_power_flow_rect(
            b,
            q[b.t_idx],
            vr[b.f_bus],
            vr[b.t_bus],
            vim[b.f_bus],
            vim[b.t_bus],
        ) for b in data.branch
    )

    @add_con(
        core,
        c_phase_angle_diff,
        c_phase_angle_diff_rect(
            b,
            vr[b.f_bus],
            vr[b.t_bus],
            vim[b.f_bus],
            vim[b.t_bus],
        ) for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    @add_con(
        core,
        c_active_power_balance,
        c_active_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus
    )

    @add_con(
        core,
        c_reactive_power_balance,
        c_reactive_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus
    )

    @add_con!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    @add_con!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    @add_con!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    @add_con!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    @add_con(
        core,
        c_from_thermal_limit,
        c_thermal_limit(b, p[b.f_idx], q[b.f_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    @add_con(
        core,
        c_to_thermal_limit,
        c_thermal_limit(b, p[b.t_idx], q[b.t_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    @add_con(
        core,
        c_voltage_magnitude,
        c_voltage_magnitude_rect(vr[b.i], vim[b.i]) for b in data.bus;
        lcon = data.vmin .^ 2,
        ucon = data.vmax .^ 2,
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

    vars2, cons2 = user_callback(core, vars, cons)
    model =ExaModel(core; kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)

    return model, vars, cons
end

"""
    ac_opf_model(filename; backend, T, form)

Return `ExaModel`, variables, and constraints for a static AC Optimal Power Flow (ACOPF) problem from the given file.

# Arguments
- `filename::String`: Path to the data file.
- `backend`: The solver backend to use. Default if nothing.
- `T`: The numeric type to use (default is `Float64`).
- `form`: Voltage representation, either `:polar` or `:rect`. Default is `:polar`.
- `user_callback`: User function that extends the model
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
    kwargs...,
)

    data = parse_ac_power_data(filename)
    data = convert_data(data, backend)

    if form == :polar
        return build_polar_opf(data, user_callback, backend = backend, T=T, kwargs...)
    elseif form == :rect
        return build_rect_opf(data, user_callback, backend = backend, T=T, kwargs...)
    else
        error("Invalid coordinate symbol - valid options are :polar or :rect")
    end
end
