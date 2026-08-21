# PowerIO returns the general GOC3 topology and time-series rows, keyed by uid
# and per-class document-order ordinals (j_dev within the class, j_sdd in the
# canonical producers-then-consumers stacking). The stacked global variable
# numbering used below (j, j_pr, j_cs, j_prcs, j_sh: offsets into one variable
# vector) is specific to this optimization model, so it is threaded on here —
# as arithmetic on the ordinals, never as a read of a uid spelling.

# Shutdown power capability p_sdpc[j_prcs, t, t_prime]. Model-specific: it is
# indexed in this model's stacked producer/consumer variable space.
#
# `devices` are the threaded producer and consumer rows, which carry PowerIO's
# typed device fields: `p_0` is the GOC3 `initial_status.p`, `p_rd_sd` is
# `p_shutdown_ramp_ub`, and `p_min` is the `p_lb` time series.
function goc3_shutdown_power_cap(devices, lengths, periods, a_start, a_end)
    p_sdpc = zeros(lengths.L_J_cspr, lengths.L_T, lengths.L_T)
    for d in devices
        for t in periods, t_prime in periods
            t >= t_prime || continue
            base = t_prime == 1 ? d.p_0 : d.p_min[t_prime-1]
            p_sdpc[d.j_prcs, t, t_prime] =
                base - d.p_rd_sd * (a_end[t] - a_start[t_prime])
        end
    end
    return p_sdpc
end

# Constraints 69 and 112/113 (and their consumer twins) need only two sums over
# the startup power capability, so the sums are accumulated directly. The row
# arrays they were built from held one entry per (device, t, t_prime) triple and
# never left this file.
#
# `commitment` maps a device uid to its unit commitment record, so the status
# lookup is a hash rather than a scan over the whole commitment table per row.
# `p_ru_su` is the GOC3 `p_startup_ramp_ub` and `p_min` the `p_lb` series.
function goc3_startup_power_cap_sums!(sum_p, sum_u, devices, commitment, periods, a_end, jsel)
    for d in devices
        rec = get(commitment, d.uid, nothing)
        rec === nothing && continue
        u_su = rec["su_status"]
        jc = jsel(d)
        for t in periods, t_prime in periods
            t_prime > t || continue
            p_supc = d.p_min[t_prime] - d.p_ru_su * (a_end[t_prime] - a_end[t])
            p_supc > 0 || continue
            sum_p[jc, t] += p_supc * u_su[t_prime]
            sum_u[jc, t] += u_su[t_prime]
        end
    end
    return nothing
end

# Constraints 70 and 112/113, the shutdown half of the same pair.
function goc3_shutdown_power_cap_sums!(sum_p, sum_u, devices, commitment, periods, p_sdpc, jsel)
    for d in devices
        rec = get(commitment, d.uid, nothing)
        rec === nothing && continue
        u_sd = rec["sd_status"]
        jc = jsel(d)
        for t in periods, t_prime in periods
            t_prime <= t || continue
            v = p_sdpc[d.j_prcs, t, t_prime]
            v > 0 || continue
            sum_p[jc, t] += v * u_sd[t_prime]
            sum_u[jc, t] += u_sd[t_prime]
        end
    end
    return nothing
end

function parse_sc_data(scopf_text, uc_data)
    # One call to PowerIO returns the full set of format-neutral GOC3 SCOPF index
    # sets; this function threads on the model's stacked variable numbering below.
    # `scopf_text` is the GOC3 document itself (read the file at the call site).
    scd = PowerIO.goc3_scopf_data(scopf_text)
    sc_data = scd.static
    lengths = scd.lengths

    # `lengths` is PowerIO's NamedTuple of static index-set sizes, including the
    # contingency count K. Bind the names this function uses by name; the four
    # energy-window lengths are computed below and merged on before it is returned.
    (; L_J_ln, L_J_ac, L_J_br, L_J_cs, L_J_pr, L_J_cspr, L_T, L_N_p, K) = lengths

    # Thread this model's stacked global variable indices onto PowerIO's general
    # rows: arithmetic on the document-order ordinals the rows carry.
    _stack_pr(r) = (j = r.j_sdd + L_J_br, j_prcs = r.j_sdd, j_pr = r.j_dev)
    _stack_cs(r) = (j = r.j_sdd + L_J_br, j_prcs = r.j_sdd, j_cs = r.j_dev)
    sc_data = (
        bus = sc_data.bus,
        shunt = [(j = s.j_sh + L_J_br + L_J_cspr, s...) for s in sc_data.shunt],
        acl_branch = [(j = b.j_ln, j_ac = b.j_ln, b...) for b in sc_data.acl_branch],
        acx_branch = [(j = b.j_xf + L_J_ln, j_ac = b.j_xf + L_J_ln, b...) for b in sc_data.acx_branch],
        vpd = [(j = b.j_xf + L_J_ln, j_ac = b.j_xf + L_J_ln, b...) for b in sc_data.vpd],
        fpd = [(j = b.j_xf + L_J_ln, j_ac = b.j_xf + L_J_ln, b...) for b in sc_data.fpd],
        vwr = [(j = b.j_xf + L_J_ln, j_ac = b.j_xf + L_J_ln, b...) for b in sc_data.vwr],
        fwr = [(j = b.j_xf + L_J_ln, j_ac = b.j_xf + L_J_ln, b...) for b in sc_data.fwr],
        dc_branch = [(j = b.j_dc + L_J_ac, b...) for b in sc_data.dc_branch],
        prod = [(; _stack_pr(p)..., p...) for p in sc_data.prod],
        cons = [(; _stack_cs(c)..., c...) for c in sc_data.cons],
        active_reserve = [(n = r.n_p, r...) for r in sc_data.active_reserve],
        reactive_reserve = [(n = r.n_q + L_N_p, r...) for r in sc_data.reactive_reserve],
        active_reserve_set_pr = [(i = r.i, _stack_pr(r)..., n = r.n_p, n_p = r.n_p) for r in sc_data.active_reserve_set_pr],
        active_reserve_set_cs = [(i = r.i, _stack_cs(r)..., n = r.n_p, n_p = r.n_p) for r in sc_data.active_reserve_set_cs],
        reactive_reserve_set_pr = [(i = r.i, _stack_pr(r)..., n = r.n_q + L_N_p, n_q = r.n_q) for r in sc_data.reactive_reserve_set_pr],
        reactive_reserve_set_cs = [(i = r.i, _stack_cs(r)..., n = r.n_q + L_N_p, n_q = r.n_q) for r in sc_data.reactive_reserve_set_cs],
    )

    # Stacked indices by device uid, for the rows that reference a device from
    # another table (energy windows, price blocks).
    pr_stack = Dict(p.uid => (j = p.j, j_prcs = p.j_prcs, j_pr = p.j_pr) for p in sc_data.prod)
    cs_stack = Dict(c.uid => (j = c.j, j_prcs = c.j_prcs, j_cs = c.j_cs) for c in sc_data.cons)

    periods = 1:L_T
    dt = scd.dt
    # Interval end times, once, from the cumulative durations. `goc3_interval_bounds`
    # re-sums `dt[1:t]` on every call, and these are read inside a t by t_prime loop
    # over every device.
    a_end = cumsum(dt)
    a_start = a_end .- dt

    sdd_uc = uc_data["time_series_output"]["simple_dispatchable_device"]
    PowerIO.goc3_add_status_flags!(uc_data["time_series_output"]["ac_line"], scd.static.acl_branch)
    PowerIO.goc3_add_status_flags!(uc_data["time_series_output"]["two_winding_transformer"], scd.static.acx_branch)
    PowerIO.goc3_add_status_flags!(sdd_uc, vcat(scd.static.prod, scd.static.cons))
    # The commitment table is joined to the device rows by uid; as a linear scan per
    # (device, t, t_prime) triple that join was the dominant cost of this function.
    commitment = Dict{String,Any}(String(uc["uid"]) => uc for uc in sdd_uc)

    #Sums correspond to constraint 69 (p_supc*u_su) and to 112/113 and 122-126 (u_su)
    sum_T_supc_pr = zeros(L_J_pr, L_T)
    sum2_T_supc_pr = zeros(L_J_pr, L_T)
    goc3_startup_power_cap_sums!(sum_T_supc_pr, sum2_T_supc_pr, sc_data.prod, commitment,
                                 periods, a_end, p -> p.j_pr)

    sum_T_supc_cs = zeros(L_J_cs, L_T)
    sum2_T_supc_cs = zeros(L_J_cs, L_T)
    goc3_startup_power_cap_sums!(sum_T_supc_cs, sum2_T_supc_cs, sc_data.cons, commitment,
                                 periods, a_end, c -> c.j_cs)

    # p_sdpc (shutdown power capability), indexed [j_prcs, t, t_prime]
    p_sdpc = goc3_shutdown_power_cap(Iterators.flatten((sc_data.prod, sc_data.cons)),
                                     lengths, periods, a_start, a_end)

    #Sums correspond to constraint 70 (p_sdpc*u_sd) and to 112/113 and 122-126 (u_sd)
    sum_T_sdpc_pr = zeros(L_J_pr, L_T)
    sum2_T_sdpc_pr = zeros(L_J_pr, L_T)
    goc3_shutdown_power_cap_sums!(sum_T_sdpc_pr, sum2_T_sdpc_pr, sc_data.prod, commitment,
                                  periods, p_sdpc, p -> p.j_pr)

    sum_T_sdpc_cs = zeros(L_J_cs, L_T)
    sum2_T_sdpc_cs = zeros(L_J_cs, L_T)
    goc3_shutdown_power_cap_sums!(sum_T_sdpc_cs, sum2_T_sdpc_cs, sc_data.cons, commitment,
                                  periods, p_sdpc, c -> c.j_cs)
    

    # Multi-interval energy requirement windows and their per-period membership.
    ew = scd.energy_windows
    W_en_max_pr = [(w_en_max_pr_ind = r.w_en_max_pr_ind, pr_stack[r.uid]..., a_en_max_start = r.a_en_max_start, a_en_max_end = r.a_en_max_end, e_max = r.e_max) for r in ew.W_en_max_pr]
    W_en_max_cs = [(w_en_max_cs_ind = r.w_en_max_cs_ind, cs_stack[r.uid]..., a_en_max_start = r.a_en_max_start, a_en_max_end = r.a_en_max_end, e_max = r.e_max) for r in ew.W_en_max_cs]
    W_en_min_pr = [(w_en_min_pr_ind = r.w_en_min_pr_ind, pr_stack[r.uid]..., a_en_min_start = r.a_en_min_start, a_en_min_end = r.a_en_min_end, e_min = r.e_min) for r in ew.W_en_min_pr]
    W_en_min_cs = [(w_en_min_cs_ind = r.w_en_min_cs_ind, cs_stack[r.uid]..., a_en_min_start = r.a_en_min_start, a_en_min_end = r.a_en_min_end, e_min = r.e_min) for r in ew.W_en_min_cs]
    T_w_en_max_pr = [(w_en_max_pr_ind = r.w_en_max_pr_ind, pr_stack[r.uid]..., t = r.t, dt = r.dt) for r in ew.T_w_en_max_pr]
    T_w_en_max_cs = [(w_en_max_cs_ind = r.w_en_max_cs_ind, cs_stack[r.uid]..., t = r.t, dt = r.dt) for r in ew.T_w_en_max_cs]
    T_w_en_min_pr = [(w_en_min_pr_ind = r.w_en_min_pr_ind, pr_stack[r.uid]..., t = r.t, dt = r.dt) for r in ew.T_w_en_min_pr]
    T_w_en_min_cs = [(w_en_min_cs_ind = r.w_en_min_cs_ind, cs_stack[r.uid]..., t = r.t, dt = r.dt) for r in ew.T_w_en_min_cs]
    L_W_en_max_pr = length(W_en_max_pr)
    L_W_en_max_cs = length(W_en_max_cs)
    L_W_en_min_pr = length(W_en_min_pr)
    L_W_en_min_cs = length(W_en_min_cs)

    pb_pr = scd.price_blocks.producer
    pb_cs = scd.price_blocks.consumer
    p_jtm_flattened_pr = [(flat_k = r.flat_k, pr_stack[r.uid]..., t = r.t, m = r.m, c_en = r.c_en, p_max = r.p_max) for r in pb_pr]
    p_jtm_flattened_cs = [(flat_k = r.flat_k, cs_stack[r.uid]..., t = r.t, m = r.m, c_en = r.c_en, p_max = r.p_max) for r in pb_cs]

    # Post-contingency surviving AC branches. PowerIO enumerates the pure survivor
    # rows per contingency; here we attach the UC on_status and expand over
    # periods, preserving the (ctg, t, component, branch) flat ordering.
    ctg_ac_survivors = scd.ac_contingency_survivors
    uc_ln_lookup = Dict(uc["uid"] => uc for uc in uc_data["time_series_output"]["ac_line"])
    uc_xf_lookup = Dict(uc["uid"] => uc for uc in uc_data["time_series_output"]["two_winding_transformer"])

    jtk_ln_flattened = Vector{@NamedTuple{flat_jtk_ln::Int, ctg::Int, j::Int, j_ac::Int, j_ln::Int, to_bus::Int, fr_bus::Int, b_sr::Float64, s_max_ctg::Float64, u_on::Int, t::Int, dt::Float64}}()
    flat_jtk_ln = 1
    for rows in ctg_ac_survivors.ln
        for t in periods
            for r in rows
                haskey(uc_ln_lookup, r.uid) || continue
                uc = uc_ln_lookup[r.uid]
                push!(jtk_ln_flattened, (flat_jtk_ln=flat_jtk_ln, ctg=r.ctg, j=r.j_ln, j_ac=r.j_ln, j_ln=r.j_ln,
                to_bus=r.to_bus, fr_bus=r.fr_bus, b_sr=r.b_sr, s_max_ctg=r.s_max_ctg, u_on=uc["on_status"][t], t=t, dt=dt[t]))
                flat_jtk_ln += 1
            end
        end
    end

    jtk_xf_flattened = Vector{@NamedTuple{flat_jtk_xf::Int, ctg::Int, j::Int, j_ac::Int, j_xf::Int, to_bus::Int, fr_bus::Int, b_sr::Float64, s_max_ctg::Float64, u_on::Int, t::Int, dt::Float64}}()
    flat_jtk_xf = 1
    for rows in ctg_ac_survivors.xf
        for t in periods
            for r in rows
                haskey(uc_xf_lookup, r.uid) || continue
                uc = uc_xf_lookup[r.uid]
                push!(jtk_xf_flattened, (flat_jtk_xf=flat_jtk_xf, ctg=r.ctg, j=r.j_xf + L_J_ln, j_ac=r.j_xf + L_J_ln, j_xf=r.j_xf,
                to_bus=r.to_bus, fr_bus=r.fr_bus, b_sr=r.b_sr, s_max_ctg=r.s_max_ctg, u_on=uc["on_status"][t], t=t, dt=dt[t]))
                flat_jtk_xf += 1
            end
        end
    end

    jtk_dc_flattened = [(flat_jtk_dc = r.flat_jtk_dc, ctg = r.ctg, j = r.j_dc + L_J_ac, j_dc = r.j_dc, to_bus = r.to_bus, fr_bus = r.fr_bus, t = r.t, dt = r.dt) for r in scd.dc_contingency_flows]

    empty_vpd = Vector{NamedTuple{(:j, :j_ac, :j_xf, :phi_min, :phi_max, :t), Tuple{Int64, Int64, Int64, Float64, Float64, Int64}}}()
    empty_fpd = Vector{NamedTuple{(:j, :j_ac, :j_xf, :phi_o, :t), Tuple{Int64, Int64, Int64, Float64, Int64}}}()
    empty_vwr = Vector{NamedTuple{(:j, :j_ac, :j_xf, :tau_min, :tau_max, :t), Tuple{Int64, Int64, Int64, Float64, Float64, Int64}}}()
    empty_fwr = Vector{NamedTuple{(:j, :j_ac, :j_xf, :tau_o, :t), Tuple{Int64, Int64, Int64, Float64, Int64}}}()
    # Reactive capability row types. A typed comprehension yields `Vector{Row}` even
    # when the filter selects nothing, so these are the element type for both the
    # populated and the empty case; no separate empty sentinel is needed. `u_on` is
    # the UC status and the sum2_T_* lookups are Float64 by their zeros(...) sources,
    # so the declarations pin what the comprehension would otherwise infer per case.
    PrPqBoundsRow = NamedTuple{(:j, :jprcs, :j_pr, :u_on, :sum2_T_supc_pr_jt, :sum2_T_sdpc_pr_jt, :beta_max, :beta_min, :q_max_p0, :q_min_p0, :t), Tuple{Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Float64, Float64, Int64}}
    PrPqeRow = NamedTuple{(:j, :jprcs, :j_pr, :u_on, :sum2_T_supc_pr_jt, :sum2_T_sdpc_pr_jt, :beta, :q_p0, :t), Tuple{Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Int64}}
    CsPqBoundsRow = NamedTuple{(:j, :jprcs, :j_cs, :u_on, :sum2_T_supc_cs_jt, :sum2_T_sdpc_cs_jt, :beta_max, :beta_min, :q_max_p0, :q_min_p0, :t), Tuple{Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Float64, Float64, Int64}}
    CsPqeRow = NamedTuple{(:j, :jprcs, :j_cs, :u_on, :sum2_T_supc_cs_jt, :sum2_T_sdpc_cs_jt, :beta, :q_p0, :t), Tuple{Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Int64}}

    sc_time_data = (
        ;
        periods = periods,

        c_p = [scd.violation_cost.p_bus],
        c_q = [scd.violation_cost.q_bus],
        c_s = [scd.violation_cost.s],
        c_e = [scd.violation_cost.e],
        busarray = [(;b..., t=t, dt=dt[t]) for b in sc_data.bus, t in periods],
        k_busarray = [(;b..., t=t, k=k) for b in sc_data.bus, t in periods, k in 1:K],
        shuntarray = [
            (;s..., t=t, u_sh = uc["step"][t])
            for s in sc_data.shunt, t in periods
            for uc in uc_data["time_series_output"]["shunt"]
            if s.uid == uc["uid"]
                ],
        k_shuntarray = [(;b..., t=t, k=k) for b in sc_data.shunt, t in periods, k in 1:K],
       

        preservearray = [(;n=r.n, n_p=r.n_p, uid=r.uid, c_rgu=r.c_rgu, c_rgd=r.c_rgd, c_scr=r.c_scr, c_nsc=r.c_nsc, c_rru=r.c_rru, c_rrd=r.c_rrd,
        σ_rgu=r.σ_rgu, σ_rgd=r.σ_rgd, σ_scr=r.σ_scr, σ_nsc=r.σ_nsc, p_rru_min=r.p_rru_min[t], p_rrd_min=r.p_rrd_min[t],
        t=t, dt=dt[t]) for r in sc_data.active_reserve, t in periods],

        qreservearray = [(;n=q.n, n_q=q.n_q, uid=q.uid, c_qru=q.c_qru, c_qrd=q.c_qrd, q_qru_min=q.q_qru_min[t], q_qrd_min=q.q_qrd_min[t], t=t, dt = dt[t])
        for q in sc_data.reactive_reserve, t in periods],

        k_prarray = [(;j_prcs=b.j_prcs, j_pr=b.j_pr, bus = b.bus, uid=b.uid, t=t, k=k) for b in sc_data.prod, t in periods, k in 1:K],
        prarray = [(;j=p.j, j_prcs=p.j_prcs, j_pr=p.j_pr, bus=p.bus, uid=p.uid, c_on=p.c_on, c_sd=p.c_sd, c_su = p.c_su, p_ru=p.p_ru, p_rd=p.p_rd,  
        p_ru_su=p.p_ru_su, p_rd_sd=p.p_rd_sd, c_rgu=p.c_rgu[t], c_rgd=p.c_rgd[t], c_scr=p.c_scr[t], c_nsc=p.c_nsc[t], c_rru_on=p.c_rru_on[t],
        c_rru_off=p.c_rru_off[t], c_rrd_on=p.c_rrd_on[t], c_rrd_off=p.c_rrd_off[t], c_qru=p.c_qru[t], c_qrd=p.c_qrd[t], p_rgu_max=p.p_rgu_max,
        p_rgd_max=p.p_rgd_max, p_scr_max=p.p_scr_max, p_nsc_max=p.p_nsc_max, p_rru_on_max=p.p_rru_on_max, p_rru_off_max=p.p_rru_off_max, 
        p_rrd_on_max=p.p_rrd_on_max, p_rrd_off_max=p.p_rrd_off_max, p_0=p.p_0, q_0=p.q_0, p_max=p.p_max[t], p_min=p.p_min[t], q_max=p.q_max[t], q_min=p.q_min[t], #sus = p.sus,
        u_on = uc["on_status"][t], u_su = uc["su_status"][t], u_sd = uc["sd_status"][t], t=t,
        sum_T_supc_pr_jt = sum_T_supc_pr[p.j_pr, t], sum_T_sdpc_pr_jt = sum_T_sdpc_pr[p.j_pr, t], sum2_T_supc_pr_jt=sum2_T_supc_pr[p.j_pr, t], sum2_T_sdpc_pr_jt=sum2_T_sdpc_pr[p.j_pr, t], dt = dt[t])
        for p in sc_data.prod, t in periods
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        prarray_pqbounds = PrPqBoundsRow[
                            (;j = p.j,
                            jprcs=p.j_prcs,
                            j_pr=p.j_pr,
                            u_on = uc["on_status"][t],
                            sum2_T_supc_pr_jt=sum2_T_supc_pr[p.j_pr, t],
                            sum2_T_sdpc_pr_jt=sum2_T_sdpc_pr[p.j_pr, t],
                            beta_max = p.beta_ub,
                            beta_min = p.beta_lb,
                            q_max_p0 = p.q_0_ub,
                            q_min_p0 = p.q_0_lb,
                            t=t
        )
        for p in sc_data.prod, t in periods
        if p.q_bound_cap == 1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        prarray_pqe = PrPqeRow[
                            (;j = p.j,
                            jprcs=p.j_prcs,
                            j_pr=p.j_pr,
                            u_on = uc["on_status"][t],
                            sum2_T_supc_pr_jt=sum2_T_supc_pr[p.j_pr, t],
                            sum2_T_sdpc_pr_jt=sum2_T_sdpc_pr[p.j_pr, t],
                            beta = p.beta,
                            q_p0 = p.q_p0,
                            t=t
        )
        for p in sc_data.prod, t in periods
        if p.q_linear_cap == 1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        k_csarray = [(;j_prcs=b.j_prcs, j_cs=b.j_cs, bus = b.bus, uid=b.uid, t=t, k=k) for b in sc_data.cons, t in periods, k in 1:K],

        csarray = [(;j=p.j, j_prcs=p.j_prcs, j_cs=p.j_cs, bus=p.bus, uid=p.uid, c_on=p.c_on, c_sd=p.c_sd, c_su=p.c_su, p_ru=p.p_ru, p_rd=p.p_rd,  
        p_ru_su=p.p_ru_su, p_rd_sd=p.p_rd_sd, c_rgu=p.c_rgu[t], c_rgd=p.c_rgd[t], c_scr=p.c_scr[t], c_nsc=p.c_nsc[t], c_rru_on=p.c_rru_on[t],
        c_rru_off=p.c_rru_off[t], c_rrd_on=p.c_rrd_on[t], c_rrd_off=p.c_rrd_off[t], c_qru=p.c_qru[t], c_qrd=p.c_qrd[t], p_rgu_max=p.p_rgu_max,
        p_rgd_max=p.p_rgd_max, p_scr_max=p.p_scr_max, p_nsc_max=p.p_nsc_max, p_rru_on_max=p.p_rru_on_max, p_rru_off_max=p.p_rru_off_max, 
        p_rrd_on_max=p.p_rrd_on_max, p_rrd_off_max=p.p_rrd_off_max, p_0=p.p_0, q_0=p.q_0, p_max=p.p_max[t], p_min=p.p_min[t], q_max=p.q_max[t], q_min=p.q_min[t], #sus=p.sus, 
        u_on = uc["on_status"][t], u_su = uc["su_status"][t], u_sd = uc["sd_status"][t], t=t,
        sum_T_supc_cs_jt = sum_T_supc_cs[p.j_cs, t], sum_T_sdpc_cs_jt = sum_T_sdpc_cs[p.j_cs, t], sum2_T_supc_cs_jt = sum2_T_supc_cs[p.j_cs, t], sum2_T_sdpc_cs_jt = sum2_T_sdpc_cs[p.j_cs, t], dt = dt[t])
        for p in sc_data.cons, t in periods
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        csarray_pqbounds = CsPqBoundsRow[
                            (;j = p.j,
                            jprcs=p.j_prcs,
                            j_cs=p.j_cs,
                            u_on = uc["on_status"][t],
                            sum2_T_supc_cs_jt=sum2_T_supc_cs[p.j_cs, t],
                            sum2_T_sdpc_cs_jt=sum2_T_sdpc_cs[p.j_cs, t],
                            beta_max = p.beta_ub,
                            beta_min = p.beta_lb,
                            q_max_p0 = p.q_0_ub,
                            q_min_p0 = p.q_0_lb,
                            t=t
        )
        for p in sc_data.cons, t in periods
        if p.q_bound_cap == 1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        csarray_pqe = CsPqeRow[
                            (;j = p.j,
                            jprcs=p.j_prcs,
                            j_cs=p.j_cs,
                            u_on = uc["on_status"][t],
                            sum2_T_supc_cs_jt=sum2_T_supc_cs[p.j_cs, t],
                            sum2_T_sdpc_cs_jt=sum2_T_sdpc_cs[p.j_cs, t],
                            beta = p.beta,
                            q_p0 = p.q_p0,
                            t=t
        )
        for p in sc_data.cons, t in periods
        if p.q_linear_cap == 1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        acxbrancharray = [
            (;j=b.j, j_ac=b.j_ac, j_xf=b.j_xf, uid=b.uid, to_bus=b.to_bus, fr_bus=b.fr_bus, c_su=b.c_su, c_sd=b.c_sd, s_max=b.s_max, g_sr=b.g_sr, b_sr=b.b_sr, b_ch=b.b_ch,
            g_fr=b.g_fr, g_to=b.g_to, b_fr=b.b_fr, b_to=b.b_to, u_on=uc["on_status"][t], u_su=uc["su_status"][t], u_sd=uc["sd_status"][t], t=t, dt = dt[t])
            for b in sc_data.acx_branch, t in periods
            for uc in uc_data["time_series_output"]["two_winding_transformer"]
            if b.uid == uc["uid"]],

        aclbrancharray = [
            (;j=b.j, j_ac=b.j_ac, j_ln=b.j_ln, uid=b.uid, to_bus=b.to_bus, fr_bus=b.fr_bus, c_su=b.c_su, c_sd=b.c_sd, s_max=b.s_max, g_sr=b.g_sr, b_sr=b.b_sr, b_ch=b.b_ch,
            g_fr=b.g_fr, g_to=b.g_to, b_fr=b.b_fr, b_to=b.b_to, u_on=uc["on_status"][t], u_su=uc["su_status"][t], u_sd=uc["sd_status"][t], t=t, dt = dt[t])
            for b in sc_data.acl_branch, t in periods
            for uc in uc_data["time_series_output"]["ac_line"]
            if b.uid == uc["uid"]],

        fpdarray = isempty(sc_data.fpd) ? empty_fpd : [(;b..., t=t) for b in sc_data.fpd, t in periods],
        fwrarray = isempty(sc_data.fwr) ? empty_fwr : [(;b..., t=t) for b in sc_data.fwr, t in periods],
        vpdarray = isempty(sc_data.vpd) ? empty_vpd : [(;b..., t=t) for b in sc_data.vpd, t in periods],
        vwrarray = isempty(sc_data.vwr) ? empty_vwr : [(;b..., t=t) for b in sc_data.vwr, t in periods],
        dclinearray = [(;b..., t=t) for b in sc_data.dc_branch, t in periods],

        p_jt_fr_dc_max = [dc.pdc_max for dc in sc_data.dc_branch, t in periods],
        p_jt_to_dc_max = [dc.pdc_max for dc in sc_data.dc_branch, t in periods],
        q_jt_fr_dc_lvar = [dc.qdc_fr_min for dc in sc_data.dc_branch, t in periods],
        q_jt_fr_dc_uvar = [dc.qdc_fr_max for dc in sc_data.dc_branch, t in periods],
        q_jt_to_dc_lvar = [dc.qdc_to_min for dc in sc_data.dc_branch, t in periods],
        q_jt_to_dc_uvar = [dc.qdc_to_max for dc in sc_data.dc_branch, t in periods],

        preservesetarray_pr = [(;b..., t=t) for b in sc_data.active_reserve_set_pr, t in periods],
        preservesetarray_cs = [(;b..., t=t) for b in sc_data.active_reserve_set_cs, t in periods],
        qreservesetarray_pr = [(;b..., t=t) for b in sc_data.reactive_reserve_set_pr, t in periods],
        qreservesetarray_cs = [(;b..., t=t) for b in sc_data.reactive_reserve_set_cs, t in periods],

        v_lvar = repeat([b.v_min for b in sc_data.bus], 1, L_T),
        v_uvar = repeat([b.v_max for b in sc_data.bus], 1, L_T),

        W_en_max_pr=W_en_max_pr,
        W_en_max_cs=W_en_max_cs,
        T_w_en_max_pr=T_w_en_max_pr,
        T_w_en_max_cs=T_w_en_max_cs,
        W_en_min_pr=W_en_min_pr,
        W_en_min_cs=W_en_min_cs,
        T_w_en_min_pr=T_w_en_min_pr,
        T_w_en_min_cs=T_w_en_min_cs,

        tk_index = [(t=t, k=k) for t in periods, k in 1:K],

        p_jtm_flattened_pr=p_jtm_flattened_pr,
        p_jtm_flattened_cs=p_jtm_flattened_cs,
        p_jtm_pr_uvar = [b.p_max for b in p_jtm_flattened_pr],
        p_jtm_cs_uvar = [b.p_max for b in p_jtm_flattened_cs],
        jtk_ln_flattened=jtk_ln_flattened,
        jtk_xf_flattened=jtk_xf_flattened,
        jtk_dc_flattened=jtk_dc_flattened,

    )

    # Append the model-owned energy-window lengths to PowerIO's static lengths,
    # which already carry the contingency count.
    lengths = merge(lengths, (; L_W_en_min_pr, L_W_en_min_cs, L_W_en_max_pr, L_W_en_max_cs))

    return sc_time_data, lengths
end

function save_go3_solution(case_filename, uc_filename, solution_name, result, vars, lengths)
    # The writer maps every solution row back to a variable index through the
    # instance's uid tables, so a uid can spell anything. Rebuilding the
    # instance here keeps the function self-contained (one extra parse per
    # solution write).
    scd = PowerIO.goc3_scopf_data(read(case_filename, String))
    st = scd.static
    pr_index = Dict(r.uid => r.j_dev for r in st.prod)
    cs_index = Dict(r.uid => r.j_dev for r in st.cons)
    xf_index = Dict(r.uid => r.j_xf for r in st.acx_branch)
    dc_index = Dict(r.uid => r.j_dc for r in st.dc_branch)
    bus_index = Dict(r.uid => r.i for r in st.bus)

    uc_data = JSON.parsefile(uc_filename)
    (; L_T) = lengths
    #Update simple dispatchable devices
    for line in uc_data["time_series_output"]["simple_dispatchable_device"]
        uid = String(line["uid"])
        if haskey(cs_index, uid)
            #This section corresponds to consuming devices
            solution_index = cs_index[uid]
            line["p_syn_res"] = Array(solution(result, vars.p_jt_scr_cs))[solution_index,:]
            line["p_ramp_res_up_online"] = Array(solution(result, vars.p_jt_rru_on_cs))[solution_index,:]
            line["p_nsyn_res"] = zeros(L_T)
            line["p_reg_res_up"] = Array(solution(result, vars.p_jt_rgu_cs))[solution_index,:]
            line["p_ramp_res_down_online"] = Array(solution(result, vars.p_jt_rrd_on_cs))[solution_index,:]
            line["p_on"] = Array(solution(result, vars.p_jt_on_cs))[solution_index,:]
            line["q"] = Array(solution(result, vars.q_jt_cs))[solution_index,:]
            line["p_reg_res_down"] = Array(solution(result, vars.p_jt_rgd_cs))[solution_index,:]
            line["p_ramp_res_up_offline"] = zeros(L_T)
            line["q_res_down"] = Array(solution(result, vars.q_jt_qrd_cs))[solution_index,:]
            line["q_res_up"] = Array(solution(result, vars.q_jt_qru_cs))[solution_index,:]
            line["p_ramp_res_down_offline"] = Array(solution(result, vars.p_jt_rrd_off_cs))[solution_index,:]
        else
            #Producing devices
            solution_index = pr_index[uid]
            line["p_syn_res"] = Array(solution(result, vars.p_jt_scr_pr))[solution_index,:]
            line["p_ramp_res_up_online"] = Array(solution(result, vars.p_jt_rru_on_pr))[solution_index,:]
            line["p_nsyn_res"] = Array(solution(result, vars.p_jt_nsc_pr))[solution_index,:]
            line["p_reg_res_up"] = Array(solution(result, vars.p_jt_rgu_pr))[solution_index,:]
            line["p_ramp_res_down_online"] = Array(solution(result, vars.p_jt_rrd_on_pr))[solution_index,:]
            line["p_on"] = Array(solution(result, vars.p_jt_on_pr))[solution_index,:]
            line["q"] = Array(solution(result, vars.q_jt_pr))[solution_index,:]
            line["p_reg_res_down"] = Array(solution(result, vars.p_jt_rgd_pr))[solution_index,:]
            line["p_ramp_res_up_offline"] = Array(solution(result, vars.p_jt_rru_off_pr))[solution_index,:]
            line["q_res_down"] = Array(solution(result, vars.q_jt_qrd_pr))[solution_index,:]
            line["q_res_up"] = Array(solution(result, vars.q_jt_qru_pr))[solution_index,:]
            line["p_ramp_res_down_offline"] = zeros(L_T)
        end
    end
    #Update two winding transformers
    for line in uc_data["time_series_output"]["two_winding_transformer"]
        solution_index = xf_index[String(line["uid"])]
        line["tm"] = Array(solution(result, vars.τ_jt_xf))[solution_index,:]
        line["ta"] = Array(solution(result, vars.φ_jt_xf))[solution_index,:]
    end
    #ac line flows can be inferred from voltage and angle levels

    #Update DC lines
    for line in uc_data["time_series_output"]["dc_line"]
        solution_index = dc_index[String(line["uid"])]
        line["qdc_fr"] = Array(solution(result, vars.q_jt_fr_dc))[solution_index,:]
        line["pdc_fr"] = Array(solution(result, vars.p_jt_fr_dc))[solution_index,:] #pdc_to is implied from energy conservation
        line["qdc_to"] = Array(solution(result, vars.q_jt_to_dc))[solution_index,:]
    end

    #Update buses
    for line in uc_data["time_series_output"]["bus"]
        solution_index = bus_index[String(line["uid"])]
        line["vm"] = Array(solution(result, vars.v_it))[solution_index,:]
        line["va"] = Array(solution(result, vars.θ_it))[solution_index,:]
    end

    open(solution_name, "w") do io
        JSON.print(io, uc_data, 4)
    end
end
