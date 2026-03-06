
function parse_mp_power_data(filename, N, corrective_action_ratio)

    data = parse_ac_power_data(filename)

    nbus = length(data.bus)

    empty_stor = Vector{NamedTuple{(:c, :Einit, :etac, :etad, :Srating, :Zr, :Zim, :Pexts, :Qexts, :bus, :t), Tuple{Int64, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Int64, Int64}}}()

    data = (
        ;
        data...,
        branch = [(i,t) for i in data.ref_buses, t in 1:N],
        barray = [(;b, t = t) for b in data.branch, t in 1:N ]
    )

    return data
end


function p_scopf_model(
    filename;
    backend = nothing,
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)

    data = parse_ac_power_data(filename)
    data = convert_data(data, backend)

    core = ExaCore(T; backend = backend)

    contingencies = []

    va = variable(core, length(data.bus), length(contingencies);)
    vm = variable(
            core,
            length(data.bus), length(contingencies);
            start = fill!(similar(data.bus, Float64), 1.0),
            lvar = data.vmin,
            uvar = data.vmax,
        )

    pg = variable(core, length(data.gen), length(contingencies); lvar = data.pmin, uvar = data.pmax)
    qg = variable(core, length(data.gen), length(contingencies); lvar = data.qmin, uvar = data.qmax)

    p = variable(core, length(data.arc), length(contingencies); lvar = -data.rate_a, uvar = data.rate_a)
    q = variable(core, length(data.arc), length(contingencies); lvar = -data.rate_a, uvar = data.rate_a)

    o = objective(
        core, gen_cost(g, pg[g.i]) for g in data.gen)

    #Fix power on contingency branches to 0
    #This does not satisfy outed branch, but allows us to have rectangular arrays without unconstrained variables
    c_fix_p_cont = constraint(core, p[a.i] for a in data.arcarray)
    c_fix_q_cont = constraint(core, q[a.i] for a in data.arcarray)

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
