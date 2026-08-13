function parse_mp_power_data(filename, N, corrective_action_ratio, T = Float64)

    data = parse_ac_power_data(filename, T)

    nbus = length(data.bus)

    empty_stor = Vector{NamedTuple{(:c, :Einit, :etac, :etad, :Srating, :Zr, :Zim, :Pexts, :Qexts, :bus, :t), Tuple{Int64, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Int64, Int64}}}()

    data = (
        ;
        data...,
        refarray = [(i,t) for i in data.ref_buses, t in 1:N],
        barray = [(;b, t = t) for b in data.branch, t in 1:N ],
        busarray = [(;b, t = t) for b in data.bus, t in 1:N ],
        arcarray = [(;a, t = t) for a in data.arc, t in 1:N ],
        genarray = [(;g, t = t) for g in data.gen, t in 1:N ],
        storarray = isempty(data.storage) ? empty_data =  empty_stor : [(;s, t = t) for s in data.storage, t in 1:N],
        branch_rate_a = [br.rate_a for br in data.branch],
        Δp = corrective_action_ratio .* (data.pmax .- data.pmin)
    )

    return data
end

function update_load_data(busarray, curve)

    for t in eachindex(curve)
        for x in 1:size(busarray, 1)
            b = busarray[x, t]
            busarray[x, t] = (
                b=ExaPowerIO.BusData{typeof(b.b.pd)}(
                    b.b.i,
                    b.b.bus_i,
                    b.b.type,
                    b.b.pd*curve[t],
                    b.b.qd*curve[t],
                    b.b.gs*curve[t],
                    b.b.bs*curve[t],
                    b.b.area,
                    b.b.vm,
                    b.b.va,
                    b.b.baseKV,
                    b.b.zone,
                    b.b.vmax,
                    b.b.vmin,
                  ), t=t
                )
        end
    end
end

#Pd, Qd as input
function update_load_data(busarray, pd, qd, baseMVA)
    for (idx ,pd_t) in pairs(pd)
        b = busarray[idx[1], idx[2]]
        busarray[idx[1], idx[2]] = (
            b=ExaPowerIO.BusData{typeof(b.b.pd)}(
                b.b.i,
                b.b.bus_i,
                b.b.type,
                pd_t / baseMVA,
                qd[idx[1], idx[2]] / baseMVA,
                b.b.gs,
                b.b.bs,
                b.b.area,
                b.b.vm,
                b.b.va,
                b.b.baseKV,
                b.b.zone,
                b.b.vmax,
                b.b.vmin,
            ),
            t=idx[2],
        )
    end
end

#If no storage contraints, the "build_base_mpopf" returns the final version of the mpopf
# ── Multi-period: one body per formulation, as in the static case ───────────
#
# The multi-period spine is its OWN — generation, flows, objective, thermal
# limits and the ramp rate all come before the voltage block, where the static
# body puts voltage first — so this is a separate class rather than the static
# body with a time index. Within it, polar and rect differ only in the voltage
# block and in which method of each expression is called, exactly as they do
# statically, so the same `OPFForm` types dispatch here too.
#
# DC is a third formulation rather than a special case: no reactive half, flows
# per branch instead of per arc, ohms law in place of the four branch flows,
# and no thermal limits. The ramp rate is on `pg` and carries over unchanged.

function add_gen_vars_mp!(core, ::Union{Polar,Rect}, data, N)
    @add_var(core, pg, size(data.gen, 1), N; lvar = repeat(data.pmin, 1, N), uvar = repeat(data.pmax, 1, N))
    @add_var(core, qg, size(data.gen, 1), N; lvar = repeat(data.qmin, 1, N), uvar = repeat(data.qmax, 1, N))
    return core, (; pg, qg)
end

function add_gen_vars_mp!(core, ::DC, data, N)
    @add_var(core, pg, size(data.gen, 1), N; lvar = repeat(data.pmin, 1, N), uvar = repeat(data.pmax, 1, N))
    return core, (; pg)
end

function add_flow_vars_mp!(core, ::Union{Polar,Rect}, data, N)
    @add_var(core, p, size(data.arc, 1), N; lvar = repeat(-data.rate_a, 1, N), uvar = repeat(data.rate_a, 1, N))
    @add_var(core, q, size(data.arc, 1), N; lvar = repeat(-data.rate_a, 1, N), uvar = repeat(data.rate_a, 1, N))
    return core, (; p, q)
end

function add_flow_vars_mp!(core, ::DC, data, N)
    @add_var(core, pf, size(data.branch, 1), N;
        lvar = repeat(-data.branch_rate_a, 1, N), uvar = repeat(data.branch_rate_a, 1, N))
    return core, (; pf)
end

# Thermal limits need both halves of the flow, so they are AC-only.
function add_thermal_mp!(core, ::Union{Polar,Rect}, data, F)
    @add_con(core, c_from_thermal_limit,
        c_thermal_limit(b, F.p[b.f_idx, t], F.q[b.f_idx, t]) for (b, t) in data.barray;
        lcon = fill(-Inf, size(data.barray)))
    @add_con(core, c_to_thermal_limit,
        c_thermal_limit(b, F.p[b.t_idx, t], F.q[b.t_idx, t]) for (b, t) in data.barray;
        lcon = fill(-Inf, size(data.barray)))
    return core, (; c_from_thermal_limit, c_to_thermal_limit)
end
add_thermal_mp!(core, ::DC, data, F) = (core, (;))

function build_base_mpopf(core, form, data, N)
    core, G = add_gen_vars_mp!(core, form, data, N)
    core, F = add_flow_vars_mp!(core, form, data, N)

    @add_obj(core, o, gen_cost(g, G.pg[g.i, t]) for (g, t) in data.genarray)

    core, thermal = add_thermal_mp!(core, form, data, F)

    @add_con(core, c_ramp_rate,
        c_ramp(G.pg[g.i, t-1], G.pg[g.i, t]) for (g, t) in data.genarray[:, 2:N];
        lcon = repeat(-data.Δp, 1, N-1),
        ucon = repeat(data.Δp, 1, N-1))

    return core, merge(G, F), merge(thermal, (; c_ramp_rate))
end

# ── the per-formulation halves ──────────────────────────────────────────────

function add_voltage_mp!(core, ::Polar, data, Nbus, N)
    @add_var(core, va, Nbus, N; lvar = -pi, uvar = pi)
    @add_var(core, vm, Nbus, N;
        start = ones(size(data.busarray)),
        lvar = repeat(data.vmin, 1, N),
        uvar = repeat(data.vmax, 1, N))
    return core, (; va, vm)
end

function add_voltage_mp!(core, ::Rect, data, Nbus, N)
    @add_var(core, vr, Nbus, N; start = ones(size(data.busarray)))
    @add_var(core, vim, Nbus, N;)
    return core, (; vr, vim)
end

function add_voltage_mp!(core, ::DC, data, Nbus, N)
    @add_var(core, va, Nbus, N; lvar = -pi, uvar = pi)
    return core, (; va)
end

# The same expressions as the static body, read at `[i, t]` rather than `[i]`.
@inline c_ref_mp(::Union{Polar,DC}, V, i, t) = c_ref_angle_polar(V.va[i, t])
@inline c_ref_mp(::Rect, V, i, t) = c_ref_angle_rect(V.vr[i, t], V.vim[i, t])

@inline c_angle_mp(::Union{Polar,DC}, b, V, t) =
    c_phase_angle_diff_polar(b, V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline c_angle_mp(::Rect, b, V, t) =
    c_phase_angle_diff_rect(b, V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])

@inline c_bal_p_mp(::Polar, b, V, t) = c_active_power_balance_demand_polar(b, V.vm[b.i, t])
@inline c_bal_p_mp(::Rect, b, V, t) = c_active_power_balance_demand_rect(b, V.vr[b.i, t], V.vim[b.i, t])
@inline c_bal_p_mp(::DC, b, V, t) = c_active_power_balance_dc(b)
@inline c_bal_q_mp(::Polar, b, V, t) = c_reactive_power_balance_demand_polar(b, V.vm[b.i, t])
@inline c_bal_q_mp(::Rect, b, V, t) = c_reactive_power_balance_demand_rect(b, V.vr[b.i, t], V.vim[b.i, t])

@inline ac_flow_mp(f, s::Symbol, b, F, V, t) = ac_flow_mp(f, Val(s), b, F, V, t)
@inline ac_flow_mp(::Polar, ::Val{:ta}, b, F, V, t) = c_to_active_power_flow_polar(b, F.p[b.f_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:tr}, b, F, V, t) = c_to_reactive_power_flow_polar(b, F.q[b.f_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:fa}, b, F, V, t) = c_from_active_power_flow_polar(b, F.p[b.t_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Polar, ::Val{:fr}, b, F, V, t) = c_from_reactive_power_flow_polar(b, F.q[b.t_idx, t], V.vm[b.f_bus, t], V.vm[b.t_bus, t], V.va[b.f_bus, t], V.va[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:ta}, b, F, V, t) = c_to_active_power_flow_rect(b, F.p[b.f_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:tr}, b, F, V, t) = c_to_reactive_power_flow_rect(b, F.q[b.f_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:fa}, b, F, V, t) = c_from_active_power_flow_rect(b, F.p[b.t_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])
@inline ac_flow_mp(::Rect, ::Val{:fr}, b, F, V, t) = c_from_reactive_power_flow_rect(b, F.q[b.t_idx, t], V.vr[b.f_bus, t], V.vr[b.t_bus, t], V.vim[b.f_bus, t], V.vim[b.t_bus, t])

function add_flow_cons_mp!(core, form::Union{Polar,Rect}, data, V, F)
    @add_con(core, c_to_active_power_flow, ac_flow_mp(form, :ta, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_to_reactive_power_flow, ac_flow_mp(form, :tr, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_from_active_power_flow, ac_flow_mp(form, :fa, b, F, V, t) for (b, t) in data.barray)
    @add_con(core, c_from_reactive_power_flow, ac_flow_mp(form, :fr, b, F, V, t) for (b, t) in data.barray)
    return core, (; c_to_active_power_flow, c_to_reactive_power_flow,
                    c_from_active_power_flow, c_from_reactive_power_flow)
end

function add_flow_cons_mp!(core, ::DC, data, V, F)
    @add_con(core, c_ohms_law,
        c_ohms_law_dcopf(br, F.pf[br.i, t], V.va[br.f_bus, t], V.va[br.t_bus, t])
        for (br, t) in data.barray)
    return core, (; c_ohms_law)
end

# The balance ROWS and the appends into them are separate steps because the
# rectangular form inserts its voltage-magnitude rows BETWEEN them. Appending a
# term adds Jacobian entries, so doing it before those rows exist puts the COO
# triplets in a different order — same model, different ordering, which the
# equivalence check sees. (The static body has the opposite order and is split
# the same way for the same reason.)
function add_balance_cons_mp!(core, form::Union{Polar,Rect}, data, V)
    @add_con(core, c_active_power_balance, c_bal_p_mp(form, b, V, t) for (b, t) in data.busarray)
    @add_con(core, c_reactive_power_balance, c_bal_q_mp(form, b, V, t) for (b, t) in data.busarray)
    return core, (; c_active_power_balance, c_reactive_power_balance)
end

function add_balance_cons_mp!(core, form::DC, data, V)
    @add_con(core, c_active_power_balance, c_bal_p_mp(form, b, V, t) for (b, t) in data.busarray)
    return core, (; c_active_power_balance)
end

function add_balance_appends_mp!(core, ::Union{Polar,Rect}, data, Nbus, B, G, F)
    c_active_power_balance, c_reactive_power_balance = B.c_active_power_balance, B.c_reactive_power_balance
    @add_con!(core, c_active_power_balance, a.bus + Nbus*(t-1) => F.p[a.i, t] for (a, t) in data.arcarray)
    @add_con!(core, c_reactive_power_balance, a.bus + Nbus*(t-1) => F.q[a.i, t] for (a, t) in data.arcarray)
    @add_con!(core, c_active_power_balance, g.bus + Nbus*(t-1) => -G.pg[g.i, t] for (g, t) in data.genarray)
    @add_con!(core, c_reactive_power_balance, g.bus + Nbus*(t-1) => -G.qg[g.i, t] for (g, t) in data.genarray)
    return core
end

function add_balance_appends_mp!(core, ::DC, data, Nbus, B, G, F)
    c_active_power_balance = B.c_active_power_balance
    @add_con!(core, c_active_power_balance, g.bus + Nbus*(t-1) => -G.pg[g.i, t] for (g, t) in data.genarray)
    @add_con!(core, c_active_power_balance, br.f_bus + Nbus*(t-1) => F.pf[br.i, t] for (br, t) in data.barray)
    @add_con!(core, c_active_power_balance, br.t_bus + Nbus*(t-1) => -F.pf[br.i, t] for (br, t) in data.barray)
    return core
end

function add_extras_mp!(core, ::Rect, data, N, V)
    @add_con(core, c_voltage_magnitude,
        c_voltage_magnitude_rect(V.vr[b.i, t], V.vim[b.i, t]) for (b, t) in data.busarray;
        lcon = repeat(data.vmin, 1, N).^2,
        ucon = repeat(data.vmax, 1, N).^2)
    return core, (; c_voltage_magnitude)
end
add_extras_mp!(core, ::Union{Polar,DC}, data, N, V) = (core, (;))

function add_mpopf_cons(core, form, data, N, Nbus, vars, cons)
    core, V = add_voltage_mp!(core, form, data, Nbus, N)
    @add_con(core, c_ref_angle, c_ref_mp(form, V, i, t) for (i, t) in data.refarray)
    core, flowcons = add_flow_cons_mp!(core, form, data, V, vars)
    @add_con(core, c_phase_angle_diff, c_angle_mp(form, b, V, t) for (b, t) in data.barray;
        lcon = repeat(data.angmin, 1, N),
        ucon = repeat(data.angmax, 1, N))
    core, balcons = add_balance_cons_mp!(core, form, data, V)
    core, extras = add_extras_mp!(core, form, data, N, V)
    core = add_balance_appends_mp!(core, form, data, Nbus, balcons, vars, vars)
    return core, merge(vars, V),
           merge(cons, (; c_ref_angle), flowcons, (; c_phase_angle_diff), balcons, extras)
end

function build_mpopf(data, Nbus, N, form, user_callback; backend = nothing, T = Float64, storage_complementarity_constraint = false, kwargs...)
    core = ExaCore(T; backend = backend)

    form = opf_form(form)
    core, vars, cons = build_base_mpopf(core, form, data, N)
    core, vars, cons = add_mpopf_cons(core, form, data, N, Nbus, vars, cons)

    if length(data.storarray) > 0
        core, vars, cons = build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)
        core, vars, cons = add_piecewise_cons(core, data, N, vars, cons, storage_complementarity_constraint, form)
    end

    core, vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; prod = true, kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)
    return model, vars, cons
end

#different constraints used when a function is added to remove complementarity and make charge/discharge curve smooth
function build_mpopf(data, Nbus, N, discharge_func::Function, form, user_callback; backend = nothing, T = Float64, kwargs...)
    core = ExaCore(T; backend = backend)

    form = opf_form(form)
    core, vars, cons = build_base_mpopf(core, form, data, N)
    core, vars, cons = add_mpopf_cons(core, form, data, N, Nbus, vars, cons)

    if length(data.storarray) > 0
        core, vars, cons = build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)
        core, vars, cons = add_smooth_cons(core, data, N, vars, cons, discharge_func, form)
    end

    core, vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; prod = true, kwargs...)

    vars = (;vars..., vars2...)
    cons = (;cons..., cons2...)
    return model, vars, cons
end

# DC storage. The AC block models the CONVERTER — reactive injection `qint`,
# current magnitude `I2`, and the ohms relation tying them to the voltage —
# none of which the DC linearization has. The STORAGE itself is active power
# and carries over: `pst` into the bus, `E` as the state, `pstd`/`pstc` in and
# out. The `pst^2 + qst^2 <= rating^2` transfer limit collapses to a bound on
# `pst`, so it is a variable bound here rather than a constraint row.
function build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form::DC)
    @add_var(core, pst, size(data.storage, 1), N;
        lvar = -repeat(data.srating, 1, N), uvar = repeat(data.srating, 1, N))
    @add_var(core, E, size(data.storage, 1), N;
        lvar = zeros(size(data.storarray)), uvar = repeat(data.emax, 1, N))
    @add_var(core, pstd, size(data.storage, 1), N; uvar = repeat(data.pdmax, 1, N))
    vars = (; vars..., pst = pst, E = E, pstd = pstd)

    c_active_power_balance = cons.c_active_power_balance
    @add_con!(core, c_active_power_balance,
        s.storage_bus + Nbus*(t-1) => pst[s.i, t] for (s, t) in data.storarray)

    return core, vars, cons
end

function build_mpopf_stor_main(core, data, N, Nbus, vars, cons, form)

    #Storage specific variables

    #active/reactive power from bus into storage
    @add_var(core, pst, size(data.storage, 1), N)
    @add_var(core, qst, size(data.storage, 1), N)

    #current magnitude squared
    @add_var(core, I2, size(data.storage, 1), N; lvar = zeros(size(data.storarray)))

    #ability of converter to control generation/absorption of reactive power
    @add_var(core, qint, size(data.storage, 1), N; lvar = -repeat(data.srating, 1, N), uvar = repeat(data.srating, 1, N))

    #energy/ state of charge
    @add_var(core, E, size(data.storage, 1), N; lvar = zeros(size(data.storarray)), uvar = repeat(data.emax, 1, N))

    #discharge from battery to grid
    @add_var(core, pstd, size(data.storage, 1), N; uvar = repeat(data.pdmax, 1, N))
    vars = (;vars..., pst=pst, qst=qst, I2=I2, qint=qint, E=E, pstd=pstd)

    c_active_power_balance = cons.c_active_power_balance
    c_reactive_power_balance = cons.c_reactive_power_balance

    @add_con!(core, c_active_power_balance, s.storage_bus + Nbus*(t-1) => pst[s.i, t] for (s, t) in data.storarray)
    @add_con!(core, c_reactive_power_balance, s.storage_bus + Nbus*(t-1) => qst[s.i, t] for (s, t) in data.storarray)

    @add_con(core, c_reactive_storage_power, c_reactive_stor_power(s, qst[s.i, t], qint[s.i, t], I2[s.i, t]) for (s, t) in data.storarray)

    @add_con(core, c_storage_transfer_thermal_limit, c_transfer_lim(s, pst[s.i, t], qst[s.i, t]) for (s, t) in data.storarray; lcon = fill(-Inf, size(data.storarray)))

    if form isa Polar
        vm = vars.vm
        @add_con(core, c_ohms, c_ohms_polar(pst[s.i, t], qst[s.i, t], vm[s.storage_bus, t], I2[s.i, t]) for (s, t) in data.storarray)
    elseif form isa Rect
        vr = vars.vr
        vim = vars.vim
        @add_con(core, c_ohms, c_ohms_rect(pst[s.i, t], qst[s.i, t], vr[s.storage_bus, t], vim[s.storage_bus, t], I2[s.i, t]) for (s, t) in data.storarray)
    end

    cons = (;cons..., c_reactive_storage_power = c_reactive_storage_power, c_storage_transfer_thermal_limit = c_storage_transfer_thermal_limit, c_ohms=c_ohms)
    return core, vars, cons
end

function add_stor_power_con!(core, ::Union{Polar,Rect}, data, V, pstc)
    @add_con(core, c_active_storage_power,
        c_active_stor_power(s, V.pst[s.i, t], V.pstd[s.i, t], pstc[s.i, t], V.I2[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_con!(core, ::DC, data, V, pstc)
    @add_con(core, c_active_storage_power,
        c_active_stor_power_dc(s, V.pst[s.i, t], V.pstd[s.i, t], pstc[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_smooth_con!(core, ::Union{Polar,Rect}, data, V)
    @add_con(core, c_active_storage_power,
        c_active_storage_power_smooth(s, V.pst[s.i, t], V.pstd[s.i, t], V.I2[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_stor_power_smooth_con!(core, ::DC, data, V)
    @add_con(core, c_active_storage_power,
        c_active_storage_power_smooth_dc(s, V.pst[s.i, t], V.pstd[s.i, t])
        for (s, t) in data.storarray)
    return core, c_active_storage_power
end

function add_piecewise_cons(core, data, N, vars, cons, storage_complementarity_constraint, form)
    #charge from battery to grid
    @add_var(core, pstc, size(data.storage, 1), N; lvar = zeros(size(data.storarray)), uvar = repeat(data.pcmax, 1, N))
    vars = (;vars..., pstc=pstc)

    pst = vars.pst
    pstd = vars.pstd
    E = vars.E

    core, c_active_storage_power = add_stor_power_con!(core, form, data, vars, pstc)

    @add_con(core, c_storage_state, c_stor_state(s, E[s.i, t], E[s.i, t - 1], pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray[:, 2:N])

    @add_con(core, c_storage_state_init, c_stor_state(s, E[s.i, t], s.energy, pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray[:, 1])

    @add_con(core, c_discharge_thermal_limit, c_discharge_lim(pstd[s.i, t], pstc[s.i, t]) for (s, t) in data.storarray; lcon = -repeat(data.srating, 1, N), ucon = repeat(data.srating, 1, N))

    @add_con(core, c_discharge_positivity, pstd[s.i, t] for (s, t) in data.storarray; ucon = fill(Inf, size(data.storarray)))

    #Complimentarity constraint
    if storage_complementarity_constraint
        @add_con(core, c_complementarity, c_comp(pstc[s.i, t], pstd[s.i, t]) for (s, t) in data.storarray)
        cons = (;cons..., c_complementarity = c_complementarity)
    end

    cons = (;cons...,
                c_active_storage_power = c_active_storage_power,
                c_storage_state = c_storage_state,
                c_storage_state_init = c_storage_state_init,
                c_discharge_thermal_limit = c_discharge_thermal_limit)

    return core, vars, cons
end

function add_smooth_cons(core, data, N, vars, cons, discharge_func, form)

    pst = vars.pst
    pstd = vars.pstd
    E = vars.E

    core, c_active_storage_power = add_stor_power_smooth_con!(core, form, data, vars)

    @add_con(core, c_storage_state, c_storage_state_smooth(s, E[s.i, t], E[s.i, t - 1], discharge_func, pstd[s.i, t]) for (s, t) in data.storarray[:, 2:N])

    @add_con(core, c_storage_state_init, c_storage_state_smooth(s, E[s.i, t], s.energy, discharge_func, pstd[s.i, t]) for (s, t) in data.storarray[:, 1])

    @add_con(core, c_discharge_thermal_limit, c_discharge_limit_smooth(pstd[s.i, t]) for (s, t) in data.storarray; lcon = -repeat(data.srating, 1, N), ucon = repeat(data.srating, 1, N))

    @add_con(core, c_charge_limit, pstd[s.i, t] for (s, t) in data.storarray; lcon = -repeat(data.pcmax, 1, N), ucon = fill(Inf, size(data.storarray)))

    cons = (;cons...,
                c_active_storage_power = c_active_storage_power,
                c_storage_state = c_storage_state,
                c_storage_state_init = c_storage_state_init,
                c_discharge_thermal_limit = c_discharge_thermal_limit,
                c_charge_limit = c_charge_limit)

    return core, vars, cons
end

"""
    mpopf_model(filename, curve; kwargs...)
    mpopf_model(filename, active_power_data, reactive_power_data; kwargs...)
    mpopf_model(filename, curve, discharge_func::Function; kwargs...)
    mpopf_model(filename, active_power_data, reactive_power_data, discharge_func::Function; kwargs...)

Construct a multi-period AC optimal power flow (MPOPF) model using different formats of load input data.

# Arguments

- `filename::String`: Path to the network data file (e.g., MATPOWER).
- `curve::AbstractVector`: A time series of demand multiplier values.
- `active_power_data::String`: Path to a matrix of active power loads (Pd) per bus and time.
- `reactive_power_data::String`: Path to a matrix of reactive power loads (Qd).
- `discharge_func::Function`: (Optional) A function specifying battery discharge losses.

## Keyword Arguments

- `N::Int`: Number of time periods (inferred if not provided).
- `corrective_action_ratio::Float64`: Ratio of corrective power action allowed (default = 0.1).
- `backend`: Optimization solver backend (deault = nothing).
- `form::Symbol`: Power flow formulation, either `:polar` or `:rect` (default = `:polar`).
- `T::Type`: Floating-point type for numeric variables (default = `Float64`).
- `storage_complementarity_constraint::Bool`: Whether to enforce complementarity for storage (only for some methods, default = false).
- `user_callback`: User function that extends the model
- `kwargs...`: Additional arguments passed to the solver or builder.

# Returns

A vector `(model::ExaModel object, variables::NamedTuple of variables, constraints::NamedTuple of constraints)` representing the MPOPF model.

# Method Variants

This function is overloaded for different combinations of input:

1. `mpopf_model(filename, curve)`
2. `mpopf_model(filename, active_power_data, reactive_power_data)`
3. `mpopf_model(filename, curve, discharge_func)`
4. `mpopf_model(filename, active_power_data, reactive_power_data, discharge_func)`
"""
function mpopf_model(
    filename, curve;
    N = length(curve),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form = :polar,
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)

    @assert length(curve) > 0
    data = parse_mp_power_data(filename, N, corrective_action_ratio, T)
    update_load_data(data.busarray, curve)
    data = convert_data(data,backend)
    Nbus = size(data.bus, 1)

    form = opf_form(form)
    return build_mpopf(data, Nbus, N, form, user_callback, backend = backend, T = T, storage_complementarity_constraint = storage_complementarity_constraint, kwargs...)

end

function mpopf_model(
    filename, active_power_data, reactive_power_data;
    pd = readdlm(active_power_data),
    qd = readdlm(reactive_power_data),
    N = size(pd, 2),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form = :polar,
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)

    data = parse_mp_power_data(filename, N, corrective_action_ratio, T)
    update_load_data(data.busarray, pd, qd, data.baseMVA[])
    data = convert_data(data,backend)
    Nbus = size(data.bus, 1)
    @assert Nbus == size(pd, 1)

    form = opf_form(form)
    return build_mpopf(data, Nbus, N, form, user_callback, backend = backend, T = T, storage_complementarity_constraint = storage_complementarity_constraint, kwargs...)

end

#Input to discharge_func should be discharge rate (or negative charge), output should be loss in battery level
function mpopf_model(
    filename, curve, discharge_func::Function;
    N = length(curve),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form = :polar,
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)

    @assert length(curve) > 0
    data = parse_mp_power_data(filename, N, corrective_action_ratio, T)
    update_load_data(data.busarray, curve)
    data = convert_data(data,backend)
    Nbus = size(data.bus, 1)

    form = opf_form(form)
    return build_mpopf(data, Nbus, N, discharge_func, form,user_callback, backend = backend, T = T, kwargs...)

end

function mpopf_model(
    filename, active_power_data, reactive_power_data, discharge_func::Function;
    pd = readdlm(active_power_data),
    qd = readdlm(reactive_power_data),
    N = size(pd, 2),
    corrective_action_ratio = 0.1,
    backend = nothing,
    form = :polar,
    T = Float64,
    storage_complementarity_constraint = false,
    user_callback = dummy_extension,
    kwargs...,
)


    data = parse_mp_power_data(filename, N, corrective_action_ratio, T)
    update_load_data(data.busarray, pd, qd, data.baseMVA[])
    data = convert_data(data,backend)
    Nbus = size(data.bus, 1)
    @assert Nbus == size(pd, 1)

    form = opf_form(form)
    return build_mpopf(data, Nbus, N, discharge_func, form,user_callback, backend = backend, T = T, kwargs...)
end

