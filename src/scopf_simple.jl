# Hard-constrained N-1 security-constrained AC OPF (SCOPF).
#
# Built on the same idiom as the multi-period model (`mpopf.jl`): the base AC OPF is
# replicated across a second dimension. Here that dimension indexes *scenarios* — column 1
# is the base case and columns 2..K+1 are the K contingencies. A contingency trips either
# a generator (its post-contingency dispatch is fixed to zero) or a branch (its admittance
# is zeroed, forcing its flows to zero so it drops out of the network). Post-contingency
# generation follows the base dispatch plus a bounded corrective adjustment `extra`.
#
# All contingency information is baked into precomputed flat arrays so the ExaModels
# generator expressions stay branch-free and the model remains generic over {T,VT} (CPU/GPU).

# Reconstruct a branch keeping every field but replacing its thermal rating. Calls the
# all-fields inner constructor directly so the admittance coefficients are NOT recomputed.
function with_rate(b::ExaPowerIO.BranchData{T}, r) where {T}
    return ExaPowerIO.BranchData{T}(
        b.i, b.f_bus, b.t_bus, b.br_r, b.br_x, b.b_fr, b.b_to, b.g_fr, b.g_to,
        T(r), b.rate_b, b.rate_c, b.tap, b.shift, b.status, b.angmin, b.angmax,
        b.f_idx, b.t_idx,
        b.c1, b.c2, b.c3, b.c4, b.c5, b.c6, b.c7, b.c8,
    )
end

# Reconstruct a branch with its admittance coefficients zeroed (line outage). With c1..c8 = 0
# the four flow equations force the branch's arc flows to zero, so it injects nothing into the
# power balance and its thermal limit (`-rate_a^2 <= 0`) is trivially slack.
function mask_branch(b::ExaPowerIO.BranchData{T}) where {T}
    return ExaPowerIO.BranchData{T}(
        b.i, b.f_bus, b.t_bus, b.br_r, b.br_x, b.b_fr, b.b_to, b.g_fr, b.g_to,
        b.rate_a, b.rate_b, b.rate_c, b.tap, b.shift, b.status, b.angmin, b.angmax,
        b.f_idx, b.t_idx,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
    )
end

function parse_scopf_power_data(filename, contingencies, corrective_action_ratio; unlimited_rate = 1.0e4, T = Float64)

    data = parse_ac_power_data(filename, T)

    ngen = length(data.gen)
    nbranch = length(data.branch)
    K = length(contingencies)
    Ns = K + 1

    Tr = eltype(data.rate_a)
    big = Tr(unlimited_rate)

    # MATPOWER convention: rateA == 0 means "unlimited". ExaPowerIO passes the 0 through, which
    # would otherwise pin the corresponding flow variables to 0, so replace zero ratings with a
    # large finite value on both the branch thermal field and the per-arc bound vector.
    branch = [iszero(b.rate_a) ? with_rate(b, big) : b for b in data.branch]
    rate_a = Tr[iszero(r) ? big : r for r in data.rate_a]

    # Which unit each scenario trips (0 = none). Scenario 1 is the base case.
    gen_out = zeros(Int, Ns)
    branch_out = zeros(Int, Ns)
    for (k, ct) in enumerate(contingencies)
        c = k + 1
        if ct.type == :gen
            (1 <= ct.idx <= ngen) || error("generator contingency idx $(ct.idx) out of range 1:$ngen")
            gen_out[c] = ct.idx
        elseif ct.type == :branch
            (1 <= ct.idx <= nbranch) || error("branch contingency idx $(ct.idx) out of range 1:$nbranch")
            branch_out[c] = ct.idx
        else
            error("unknown contingency type $(ct.type); use :gen or :branch")
        end
    end

    # Per-(branch, scenario) array; the outaged branch is masked in its own scenario.
    barray2 = [
        branch_out[c] == b.i ? (; b = mask_branch(b), c = c) : (; b = b, c = c)
        for b in branch, c in 1:Ns
    ]

    genarray2 = [(; g = g, c = c) for g in data.gen, c in 1:Ns]
    busarray2 = [(; b = b, c = c) for b in data.bus, c in 1:Ns]
    arcarray2 = [(; a = a, c = c) for a in data.arc, c in 1:Ns]
    refarray2 = [(i, c) for i in data.ref_buses, c in 1:Ns]

    # Generation bounds per scenario; a tripped generator is fixed to zero in its scenario.
    pg_lb = repeat(data.pmin, 1, Ns); pg_ub = repeat(data.pmax, 1, Ns)
    qg_lb = repeat(data.qmin, 1, Ns); qg_ub = repeat(data.qmax, 1, Ns)
    for c in 2:Ns
        g = gen_out[c]
        if g != 0
            pg_lb[g, c] = 0; pg_ub[g, c] = 0
            qg_lb[g, c] = 0; qg_ub[g, c] = 0
        end
    end

    # Corrective adjustment `extra` >= 0: zero in the base case (column 1), capped at a fraction
    # of pmax in contingencies, and zero for a tripped generator.
    extra_lb = zeros(Tr, ngen, Ns)
    extra_ub = zeros(Tr, ngen, Ns)
    for c in 2:Ns, g in 1:ngen
        extra_ub[g, c] = corrective_action_ratio * data.pmax[g]
    end
    for c in 2:Ns
        g = gen_out[c]
        g != 0 && (extra_ub[g, c] = 0)
    end

    # Corrective coupling pg[g,c] = pg[g,1] + extra[g,c] for c >= 2, skipping tripped units.
    coupling = [(; g = data.gen[gi], c = c) for c in 2:Ns for gi in 1:ngen if gen_out[c] != gi]

    # Phase-angle-difference bounds; open the outaged branch's limit in its scenario.
    angmin2 = repeat(data.angmin, 1, Ns)
    angmax2 = repeat(data.angmax, 1, Ns)
    for c in 2:Ns
        l = branch_out[c]
        if l != 0
            angmin2[l, c] = -Inf
            angmax2[l, c] = Inf
        end
    end

    data = (;
        data...,
        branch = branch,
        rate_a = rate_a,
        barray2 = barray2,
        genarray2 = genarray2,
        busarray2 = busarray2,
        arcarray2 = arcarray2,
        refarray2 = refarray2,
        coupling = coupling,
        pg_lb = pg_lb, pg_ub = pg_ub,
        qg_lb = qg_lb, qg_ub = qg_ub,
        extra_lb = extra_lb, extra_ub = extra_ub,
        angmin2 = angmin2, angmax2 = angmax2,
    )

    return data
end

function build_base_scopf(core, data, K)
    Ns = K + 1

    # active, reactive power generated, and the bounded corrective adjustment
    pg = variable(core, size(data.gen, 1), Ns; lvar = data.pg_lb, uvar = data.pg_ub)
    qg = variable(core, size(data.gen, 1), Ns; lvar = data.qg_lb, uvar = data.qg_ub)
    extra = variable(core, size(data.gen, 1), Ns; lvar = data.extra_lb, uvar = data.extra_ub)

    # active, reactive power at each arc
    p = variable(core, size(data.arc, 1), Ns; lvar = repeat(-data.rate_a, 1, Ns), uvar = repeat(data.rate_a, 1, Ns))
    q = variable(core, size(data.arc, 1), Ns; lvar = repeat(-data.rate_a, 1, Ns), uvar = repeat(data.rate_a, 1, Ns))

    # generation cost averaged over the base case and all contingencies
    o = objective(core, gen_cost(g, pg[g.i, c]) / (K + 1) for (g, c) in data.genarray2)

    # corrective coupling: pg[g,c] - pg[g,1] - extra[g,c] == 0 (tripped units excluded)
    c_corrective = constraint(core, pg[g.i, c] - pg[g.i, 1] - extra[g.i, c] for (g, c) in data.coupling)

    c_from_thermal_limit = constraint(
        core,
        c_thermal_limit(b, p[b.f_idx, c], q[b.f_idx, c]) for (b, c) in data.barray2;
        lcon = fill(-Inf, size(data.barray2)),
    )
    c_to_thermal_limit = constraint(
        core,
        c_thermal_limit(b, p[b.t_idx, c], q[b.t_idx, c]) for (b, c) in data.barray2;
        lcon = fill(-Inf, size(data.barray2)),
    )

    vars = (pg = pg, qg = qg, extra = extra, p = p, q = q)
    cons = (
        c_corrective = c_corrective,
        c_from_thermal_limit = c_from_thermal_limit,
        c_to_thermal_limit = c_to_thermal_limit,
    )
    return vars, cons
end

function add_scopf_cons(core, data, K, Nbus, vars, cons, form)
    Ns = K + 1
    pg = vars.pg; qg = vars.qg; p = vars.p; q = vars.q

    if form == :polar
        va = variable(core, Nbus, Ns; lvar = -pi, uvar = pi)
        vm = variable(
            core,
            Nbus, Ns;
            start = ones(size(data.busarray2)),
            lvar = repeat(data.vmin, 1, Ns),
            uvar = repeat(data.vmax, 1, Ns),
        )

        c_ref_angle = constraint(core, c_ref_angle_polar(va[i, c]) for (i, c) in data.refarray2)

        c_to_active_power_flow = constraint(core, c_to_active_power_flow_polar(b, p[b.f_idx, c], vm[b.f_bus, c], vm[b.t_bus, c], va[b.f_bus, c], va[b.t_bus, c]) for (b, c) in data.barray2)
        c_to_reactive_power_flow = constraint(core, c_to_reactive_power_flow_polar(b, q[b.f_idx, c], vm[b.f_bus, c], vm[b.t_bus, c], va[b.f_bus, c], va[b.t_bus, c]) for (b, c) in data.barray2)
        c_from_active_power_flow = constraint(core, c_from_active_power_flow_polar(b, p[b.t_idx, c], vm[b.f_bus, c], vm[b.t_bus, c], va[b.f_bus, c], va[b.t_bus, c]) for (b, c) in data.barray2)
        c_from_reactive_power_flow = constraint(core, c_from_reactive_power_flow_polar(b, q[b.t_idx, c], vm[b.f_bus, c], vm[b.t_bus, c], va[b.f_bus, c], va[b.t_bus, c]) for (b, c) in data.barray2)

        c_phase_angle_diff = constraint(
            core,
            c_phase_angle_diff_polar(b, va[b.f_bus, c], va[b.t_bus, c]) for (b, c) in data.barray2;
            lcon = data.angmin2,
            ucon = data.angmax2,
        )

        c_active_power_balance = constraint(core, c_active_power_balance_demand_polar(b, vm[b.i, c]) for (b, c) in data.busarray2)
        c_reactive_power_balance = constraint(core, c_reactive_power_balance_demand_polar(b, vm[b.i, c]) for (b, c) in data.busarray2)

        cons = (; cons...,
            c_ref_angle = c_ref_angle,
            c_to_active_power_flow = c_to_active_power_flow,
            c_to_reactive_power_flow = c_to_reactive_power_flow,
            c_from_active_power_flow = c_from_active_power_flow,
            c_from_reactive_power_flow = c_from_reactive_power_flow,
            c_phase_angle_diff = c_phase_angle_diff,
            c_active_power_balance = c_active_power_balance,
            c_reactive_power_balance = c_reactive_power_balance,
        )
        vars = (; vars..., va = va, vm = vm)

    elseif form == :rect
        vr = variable(core, Nbus, Ns; start = ones(size(data.busarray2)))
        vim = variable(core, Nbus, Ns;)

        c_ref_angle = constraint(core, c_ref_angle_rect(vr[i, c], vim[i, c]) for (i, c) in data.refarray2)

        c_to_active_power_flow = constraint(core, c_to_active_power_flow_rect(b, p[b.f_idx, c], vr[b.f_bus, c], vr[b.t_bus, c], vim[b.f_bus, c], vim[b.t_bus, c]) for (b, c) in data.barray2)
        c_to_reactive_power_flow = constraint(core, c_to_reactive_power_flow_rect(b, q[b.f_idx, c], vr[b.f_bus, c], vr[b.t_bus, c], vim[b.f_bus, c], vim[b.t_bus, c]) for (b, c) in data.barray2)
        c_from_active_power_flow = constraint(core, c_from_active_power_flow_rect(b, p[b.t_idx, c], vr[b.f_bus, c], vr[b.t_bus, c], vim[b.f_bus, c], vim[b.t_bus, c]) for (b, c) in data.barray2)
        c_from_reactive_power_flow = constraint(core, c_from_reactive_power_flow_rect(b, q[b.t_idx, c], vr[b.f_bus, c], vr[b.t_bus, c], vim[b.f_bus, c], vim[b.t_bus, c]) for (b, c) in data.barray2)

        c_phase_angle_diff = constraint(
            core,
            c_phase_angle_diff_rect(b, vr[b.f_bus, c], vr[b.t_bus, c], vim[b.f_bus, c], vim[b.t_bus, c]) for (b, c) in data.barray2;
            lcon = data.angmin2,
            ucon = data.angmax2,
        )

        c_active_power_balance = constraint(core, c_active_power_balance_demand_rect(b, vr[b.i, c], vim[b.i, c]) for (b, c) in data.busarray2)
        c_reactive_power_balance = constraint(core, c_reactive_power_balance_demand_rect(b, vr[b.i, c], vim[b.i, c]) for (b, c) in data.busarray2)

        c_voltage_magnitude = constraint(
            core,
            c_voltage_magnitude_rect(vr[b.i, c], vim[b.i, c]) for (b, c) in data.busarray2;
            lcon = repeat(data.vmin, 1, Ns) .^ 2,
            ucon = repeat(data.vmax, 1, Ns) .^ 2,
        )

        cons = (; cons...,
            c_ref_angle = c_ref_angle,
            c_to_active_power_flow = c_to_active_power_flow,
            c_to_reactive_power_flow = c_to_reactive_power_flow,
            c_from_active_power_flow = c_from_active_power_flow,
            c_from_reactive_power_flow = c_from_reactive_power_flow,
            c_phase_angle_diff = c_phase_angle_diff,
            c_active_power_balance = c_active_power_balance,
            c_reactive_power_balance = c_reactive_power_balance,
            c_voltage_magnitude = c_voltage_magnitude,
        )
        vars = (; vars..., vr = vr, vim = vim)
    end

    c_active_power_balance = cons.c_active_power_balance
    c_reactive_power_balance = cons.c_reactive_power_balance

    # Accumulate arc flows and generation into the per-bus, per-scenario balance rows. The
    # scenario index folds into the flattened constraint row via the Nbus*(c-1) offset.
    constraint!(core, c_active_power_balance, a.bus + Nbus * (c - 1) => p[a.i, c] for (a, c) in data.arcarray2)
    constraint!(core, c_reactive_power_balance, a.bus + Nbus * (c - 1) => q[a.i, c] for (a, c) in data.arcarray2)

    constraint!(core, c_active_power_balance, g.bus + Nbus * (c - 1) => -pg[g.i, c] for (g, c) in data.genarray2)
    constraint!(core, c_reactive_power_balance, g.bus + Nbus * (c - 1) => -qg[g.i, c] for (g, c) in data.genarray2)

    return vars, cons
end

function build_scopf(data, Nbus, K, form, user_callback; backend = nothing, T = Float64, kwargs...)
    core = ExaCore(T; backend = backend)

    vars, cons = build_base_scopf(core, data, K)
    vars, cons = add_scopf_cons(core, data, K, Nbus, vars, cons, form)

    vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; kwargs...)

    vars = (; vars..., vars2...)
    cons = (; cons..., cons2...)
    return model, vars, cons
end

"""
    scopf_model(filename, contingencies; kwargs...)

Construct a hard-constrained N-1 security-constrained AC optimal power flow (SCOPF) model.

The base AC OPF is replicated over scenarios: scenario 1 is the base case and each entry of
`contingencies` adds one post-contingency scenario. Post-contingency active generation equals
the base dispatch plus a non-negative corrective adjustment bounded by
`corrective_action_ratio * pmax`; a tripped unit is fixed to zero. Full AC power balance,
branch flow equations and apparent-power limits, phase-angle limits, and the reference-bus
angle are enforced in every scenario.

# Arguments
- `filename::String`: Path to the network data file (MATPOWER `.m`).
- `contingencies::Vector{<:NamedTuple}`: each `(type = :gen, idx = i)` (generator outage) or
  `(type = :branch, idx = l)` (line outage). `idx` is the 1-based index into the parsed
  `data.gen` / `data.branch` (i.e. matpower row order).

# Keyword Arguments
- `backend`: ExaModels backend (default `nothing`, CPU).
- `form::Symbol`: `:polar` or `:rect` (default `:polar`).
- `T::Type`: numeric type (default `Float64`).
- `corrective_action_ratio`: corrective redispatch cap as a fraction of `pmax` (default `0.05`).
- `unlimited_rate`: value (p.u.) substituted for a branch's `rate_a` when the data lists it as
  `0` (MATPOWER "unlimited"), so the flow is not pinned to zero (default `1.0e4`).
- `user_callback`: function `(core, vars, cons) -> (vars2, cons2)` extending the model.
- `kwargs...`: forwarded to `ExaModel`.

# Returns
`(model::ExaModel, vars::NamedTuple, cons::NamedTuple)`.
"""
function scopf_model(
    filename, contingencies;
    backend = nothing,
    form = :polar,
    T = Float64,
    corrective_action_ratio = 0.05,
    unlimited_rate = 1.0e4,
    user_callback = dummy_extension,
    kwargs...,
)
    if form != :polar && form != :rect
        error("Invalid coordinate symbol - valid options are :polar or :rect")
    end

    data = parse_scopf_power_data(filename, contingencies, corrective_action_ratio; unlimited_rate = unlimited_rate, T = T)
    K = length(contingencies)
    data = convert_data(data, backend)
    Nbus = size(data.bus, 1)

    return build_scopf(data, Nbus, K, form, user_callback; backend = backend, T = T, kwargs...)
end
