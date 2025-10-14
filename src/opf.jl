function dummy_extension(core, vars, cons)
    return (;), (;)
end

function build_polar_opf(data, user_callback; backend = nothing, T=Float64, kwargs...)
    core = ExaCore(T; backend = backend)

    va = variable(core, length(data.bus);)
    vm = variable(
            core,
            length(data.bus);
            start = fill!(similar(data.bus, Float64), 1.0),
            lvar = data.vmin,
            uvar = data.vmax,
        )

    pg = variable(core, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    qg = variable(core, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    p = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    q = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    o = objective(
        core, gen_cost(g, pg[g.i]) for g in data.gen)

    c_ref_angle = constraint(core, c_ref_angle_polar(va[i]) for i in data.ref_buses)

    c_to_active_power_flow = constraint(core, c_to_active_power_flow_polar(b, p[b.f_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    c_to_reactive_power_flow = constraint(core, c_to_reactive_power_flow_polar(b, q[b.f_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    c_from_active_power_flow = constraint(core, c_from_active_power_flow_polar(b, p[b.t_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    c_from_reactive_power_flow = constraint(core, c_from_reactive_power_flow_polar(b, q[b.t_idx],
        vm[b.f_bus],vm[b.t_bus],va[b.f_bus],va[b.t_bus]) for b in data.branch)

    c_phase_angle_diff = constraint(
        core,
        c_phase_angle_diff_polar(b,va[b.f_bus],va[b.t_bus]) for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    c_active_power_balance = constraint(core, c_active_power_balance_demand_polar(b, vm[b.i]) for b in data.bus)

    c_reactive_power_balance = constraint(core, c_reactive_power_balance_demand_polar(b, vm[b.i]) for b in data.bus)

    c_active_power_balance_arcs = constraint!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    c_reactive_power_balance_arcs = constraint!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    c_active_power_balance_gen = constraint!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    c_active_power_balance_gen = constraint!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    c_from_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.f_idx],q[b.f_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
        )

    c_to_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.t_idx],q[b.t_idx])
        for b in data.branch;
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
    core = ExaCore(T; backend = backend)

    vr = variable(core, length(data.bus), start = fill!(similar(data.bus, Float64), 1.0))
    vim = variable(core, length(data.bus))

    pg = variable(core, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    qg = variable(core, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    p = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    q = variable(core, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    o = objective(
        core, gen_cost(g, pg[g.i]) for g in data.gen)

    c_ref_angle = constraint(core, c_ref_angle_rect(vr[i], vim[i]) for i in data.ref_buses)

    c_to_active_power_flow = constraint(core, c_to_active_power_flow_rect(b,p[b.f_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    c_to_reactive_power_flow = constraint(core, c_to_reactive_power_flow_rect(b,q[b.f_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    c_from_active_power_flow = constraint(core, c_from_active_power_flow_rect(b,p[b.t_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)
    
    c_from_reactive_power_flow = constraint(core, c_from_reactive_power_flow_rect(b,q[b.t_idx],
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus]) for b in data.branch)

    c_phase_angle_diff = constraint(
        core, c_phase_angle_diff_rect(b,
        vr[b.f_bus],vr[b.t_bus],vim[b.f_bus],vim[b.t_bus])
        for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    c_active_power_balance = constraint(core, c_active_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus)

    c_reactive_power_balance = constraint(core, c_reactive_power_balance_demand_rect(b, vr[b.i], vim[b.i]) for b in data.bus)

    c_active_power_balance_arcs = constraint!(core, c_active_power_balance, a.bus => p[a.i] for a in data.arc)
    c_reactive_power_balance_arcs = constraint!(core, c_reactive_power_balance, a.bus => q[a.i] for a in data.arc)

    c_active_power_balance_gen = constraint!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    c_reactive_power_balance_gen = constraint!(core, c_reactive_power_balance, g.bus => -qg[g.i] for g in data.gen)

    c_from_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.f_idx], q[b.f_idx]) for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    c_to_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.t_idx], q[b.t_idx])
        for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    c_voltage_magnitude = constraint(
        core, c_voltage_magnitude_rect(vr[b.i], vim[b.i]) for b in data.bus; 
        lcon = data.vmin.^2, 
        ucon = data.vmax.^2
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
    opf_model(filename; backend, T, form)

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
function opf_model(
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
