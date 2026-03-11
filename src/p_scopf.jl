
function parse_sc_power_data(filename, contingencies)

    data = parse_ac_power_data(filename)

    nbus = length(data.bus)

    empty_stor = Vector{NamedTuple{(:c, :Einit, :etac, :etad, :Srating, :Zr, :Zim, :Pexts, :Qexts, :bus, :t), Tuple{Int64, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Int64, Int64}}}()

    K = length(contingencies)+1

    n_branch = length(data.branch)

    data = (
        ;
        data...,
        refarray = [(i,k) for i in data.ref_buses, k in 1:K],
        busarray = [(;b, k = k) for b in data.bus, k in 1:K ],
        genarray = [(;g, k = k) for g in data.gen, k in 1:K ],
        brancharray = [(;b, k = k) for b in data.branch, k in 1:K if k == 1 || b.i != contingencies[k-1]],
        arcarray = [(;a, k = k) for a in data.arc, k in 1:K if k == 1 || (a.i != contingencies[k-1] && a.i != n_branch + contingencies[k-1])],
        contingencyarcarray = [(;a, k = k) for a in data.arc, k in 2:K if (a.i == contingencies[k-1] || a.i == n_branch + contingencies[k-1])],
    )

    return data
end


function p_scopf_model(
    filename, contingencies_file;
    backend = nothing,
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)
    contingencies = readdlm(contingencies_file)
    
    data = parse_sc_power_data(filename, contingencies)
    data = convert_data(data, backend)

    Nbus = size(data.bus, 1)

    core = ExaCore(T; backend = backend)

    K = length(contingencies) + 1

    #use k for second index corresponging to contingency (1=base)

    va = variable(core, length(data.bus), K; lvar = -pi, uvar = pi)
    vm = variable(
            core,
            length(data.bus), K;
            start = fill!(similar(data.bus, Float64), 1.0),
            lvar = repeat(data.vmin, 1, K),
            uvar = repeat(data.vmax, 1, K),
        )

    pg = variable(core, length(data.gen), K; lvar = repeat(data.pmin, 1, K), uvar = repeat(data.pmax, 1, K))
    qg = variable(core, length(data.gen), K; lvar = repeat(data.qmin, 1, K), uvar = repeat(data.qmax, 1, K))

    p = variable(core, length(data.arc), K; lvar = repeat(-data.rate_a, 1, K), uvar = repeat(data.rate_a, 1, K))
    q = variable(core, length(data.arc), K; lvar = repeat(-data.rate_a, 1, K), uvar = repeat(data.rate_a, 1, K))

    o = objective(
        core, gen_cost(g, pg[g.i, 1]) for g in data.gen)

    #Fix power on contingency branches to 0
    #This does not satisfy outed branch, but allows us to have rectangular arrays without unconstrained variables
    c_fix_p_cont = constraint(core, p[a.i, k] for (a, k) in data.contingencyarcarray)
    c_fix_q_cont = constraint(core, q[a.i, k] for (a, k) in data.contingencyarcarray)

    println(data.contingencyarcarray)
    #Power is constant for all Pg, Vm across contingencies
    c_fix_pg_cont = constraint(core, pg[g.i, 1] - pg[g.i, k] for (g, k) in data.genarray if k > 1 && data.ref_buses[1] != g.bus)
    c_fix_vm_cont = constraint(core, vm[g.bus, 1] - vm[g.bus, k] for (g, k) in data.genarray if k > 1)# && data.ref_buses[1] != g.bus)

    
    #add con(k)
    c_ref_angle = constraint(core, c_ref_angle_polar(va[i, k]) for (i, k) in data.refarray)


    #iterate over data.branch (does not include contingencies)
    c_to_active_power_flow = constraint(core, c_to_active_power_flow_polar(b, p[b.f_idx, k],
        vm[b.f_bus, k],vm[b.t_bus, k],va[b.f_bus, k],va[b.t_bus, k]) for (b, k) in data.brancharray)

    c_to_reactive_power_flow = constraint(core, c_to_reactive_power_flow_polar(b, q[b.f_idx, k],
        vm[b.f_bus, k],vm[b.t_bus, k],va[b.f_bus, k],va[b.t_bus, k]) for (b, k) in data.brancharray)

    
    c_from_active_power_flow = constraint(core, c_from_active_power_flow_polar(b, p[b.t_idx, k],
        vm[b.f_bus, k],vm[b.t_bus, k],va[b.f_bus, k],va[b.t_bus, k]) for (b, k) in data.brancharray)

    c_from_reactive_power_flow = constraint(core, c_from_reactive_power_flow_polar(b, q[b.t_idx, k],
        vm[b.f_bus, k],vm[b.t_bus, k],va[b.f_bus, k],va[b.t_bus, k]) for (b, k) in data.brancharray)

    c_phase_angle_diff = constraint(
        core,
        c_phase_angle_diff_polar(b,va[b.f_bus, k],va[b.t_bus, k]) for (b, k) in data.brancharray;
        lcon = repeat(data.angmin, 1, K),
        ucon = repeat(data.angmax, 1, K),
    )
    
    c_active_power_balance = constraint(core, c_active_power_balance_demand_polar(b, vm[b.i, k]) for (b, k) in data.busarray)

    c_reactive_power_balance = constraint(core, c_reactive_power_balance_demand_polar(b, vm[b.i, k]) for (b, k) in data.busarray)

    #add appropriate tracking system
    c_active_power_balance_arcs = constraint!(core, c_active_power_balance, a.bus + Nbus*(k-1) => p[a.i, k] for (a, k) in data.arcarray)
    c_reactive_power_balance_arcs = constraint!(core, c_reactive_power_balance, a.bus + Nbus*(k-1) => q[a.i, k] for (a, k) in data.arcarray)

    c_active_power_balance_gen = constraint!(core, c_active_power_balance, g.bus + Nbus*(k-1) => -pg[g.i, k] for (g, k) in data.genarray)
    c_active_power_balance_gen = constraint!(core, c_reactive_power_balance, g.bus + Nbus*(k-1)=> -qg[g.i, k] for (g, k) in data.genarray)

    
    c_from_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.f_idx, k],q[b.f_idx, k]) for (b, k) in data.brancharray;
        lcon = fill(-Inf, size(data.brancharray))
    )

    
    c_to_thermal_limit = constraint(
        core, c_thermal_limit(b,p[b.t_idx, k],q[b.t_idx, k])
        for (b, k) in data.brancharray;
        lcon = fill(-Inf, size(data.brancharray))
    )
    
    vars = (
            va = va,
            vm = vm,
            pg = pg,
            qg = qg,
            p = p,        
            q = q
        )

    #=cons = (
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
    )=#
    cons=2

    #vars2, cons2 = user_callback(core, vars, cons)
    model =ExaModel(core; kwargs...)

    #vars = (;vars..., vars2...)
    #cons = (;cons..., cons2...)

    return model, vars, cons
end
