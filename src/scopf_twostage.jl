# =====================================================================
# Two-stage form of the hard-constrained N-1 SCOPF (`scopf_model`).
#
# Built on `TwoStageExaCore`. The base case is the FIRST stage (design
# variables + design constraints); each contingency is a SECOND-stage
# (per-scenario) block. The corrective coupling `pgk = pg0 + extrak`
# touches the first-stage base dispatch `pg0` plus only its own scenario's
# variables, which is the linking structure the Schur solver exploits.
#
# Scenario count = K (the number of contingencies). Unlike `scopf_model`,
# where the base case is "column 1" of K+1, here the base case is the
# design block and is NOT a scenario. Physics, masking (`mask_branch`),
# the rate_a=0 fix (`with_rate`), and the corrective recourse are identical
# to `scopf_simple.jl`. Both `:polar` and `:rect` voltage forms are supported.
# =====================================================================

function parse_scopf_twostage_data(filename, contingencies, corrective_action_ratio; unlimited_rate = 1.0e4, T = Float64)

    data = parse_ac_power_data(filename, T)

    ngen = length(data.gen)
    nbus = length(data.bus)
    narc = length(data.arc)
    nbranch = length(data.branch)
    K = length(contingencies)
    K >= 1 || error("scopf_twostage_model requires at least one contingency (K >= 1). Use scopf_model for the no-contingency case.")

    Tr = eltype(data.rate_a)
    big = Tr(unlimited_rate)

    # MATPOWER rateA == 0 means "unlimited"; replace with a large finite value (see scopf_simple.jl).
    branch = [iszero(b.rate_a) ? with_rate(b, big) : b for b in data.branch]
    rate_a = Tr[iszero(r) ? big : r for r in data.rate_a]

    # Contingency target per scenario k = 1..K (0 = none).
    gen_out = zeros(Int, K)
    branch_out = zeros(Int, K)
    for (k, ct) in enumerate(contingencies)
        if ct.type == :gen
            (1 <= ct.idx <= ngen) || error("generator contingency idx $(ct.idx) out of range 1:$ngen")
            gen_out[k] = ct.idx
        elseif ct.type == :branch
            (1 <= ct.idx <= nbranch) || error("branch contingency idx $(ct.idx) out of range 1:$nbranch")
            branch_out[k] = ct.idx
        else
            error("unknown contingency type $(ct.type); use :gen or :branch")
        end
    end

    # Per-scenario flattened arrays. Comprehensions are column-major (first index
    # fast, scenario k slow) — the scenario-slow layout the two-stage core expects.
    barray_sc = [
        branch_out[k] == b.i ? (; b = mask_branch(b), k = k) : (; b = b, k = k)
        for b in branch, k in 1:K
    ]
    busarray_sc = [(; b = b, k = k) for b in data.bus, k in 1:K]
    arcarray_sc = [(; a = a, k = k) for a in data.arc, k in 1:K]
    genarray_sc = [(; g = g, k = k) for g in data.gen, k in 1:K]
    refarray_sc = [(; i = i, k = k) for i in data.ref_buses, k in 1:K]
    coupling_sc = [(; g = g, k = k) for g in data.gen, k in 1:K]  # uniform: every gen, every scenario

    # Per-(gen, scenario) bounds; a tripped generator is fixed to zero in its scenario,
    # and its corrective `extra` is relaxed to absorb -pg0 so the (kept, uniform) coupling
    # row stays feasible without constraining the base dispatch.
    pgk_lb = repeat(data.pmin, 1, K); pgk_ub = repeat(data.pmax, 1, K)
    qgk_lb = repeat(data.qmin, 1, K); qgk_ub = repeat(data.qmax, 1, K)
    extrak_lb = zeros(Tr, ngen, K)
    extrak_ub = repeat(corrective_action_ratio .* data.pmax, 1, K)
    for k in 1:K
        g = gen_out[k]
        if g != 0
            pgk_lb[g, k] = 0; pgk_ub[g, k] = 0
            qgk_lb[g, k] = 0; qgk_ub[g, k] = 0
            extrak_lb[g, k] = -data.pmax[g]
            extrak_ub[g, k] = 0
        end
    end

    vmk_lb = repeat(data.vmin, 1, K); vmk_ub = repeat(data.vmax, 1, K)
    rate_mat = repeat(rate_a, 1, K)

    # Phase-angle-difference bounds per scenario; open the outaged branch's limit.
    angmin_sc = repeat(data.angmin, 1, K)
    angmax_sc = repeat(data.angmax, 1, K)
    for k in 1:K
        l = branch_out[k]
        if l != 0
            angmin_sc[l, k] = -Inf
            angmax_sc[l, k] = Inf
        end
    end

    data = (;
        data...,
        branch = branch,
        rate_a = rate_a,
        barray_sc = barray_sc,
        busarray_sc = busarray_sc,
        arcarray_sc = arcarray_sc,
        genarray_sc = genarray_sc,
        refarray_sc = refarray_sc,
        coupling_sc = coupling_sc,
        pgk_lb = pgk_lb, pgk_ub = pgk_ub,
        qgk_lb = qgk_lb, qgk_ub = qgk_ub,
        extrak_lb = extrak_lb, extrak_ub = extrak_ub,
        vmk_lb = vmk_lb, vmk_ub = vmk_ub,
        rate_mat = rate_mat,
        angmin_sc = angmin_sc, angmax_sc = angmax_sc,
    )

    return data, K, nbus, narc, ngen, nbranch
end

"""
    scopf_twostage_model(filename, contingencies; backend, T, form, corrective_action_ratio, unlimited_rate, user_callback, kwargs...)

Two-stage form of the hard-constrained N-1 SCOPF (`scopf_model`), built on
`TwoStageExaCore` from ExaModels.jl so it can be solved with MadNLP's
`SchurComplementKKTSystem`, which factorizes the per-contingency block structure
in parallel.

Same physics as `scopf_model` (`scopf_simple.jl`): full AC power balance, branch
flow equations + apparent-power limits, phase-angle limits, reference-bus angle,
line outages via `mask_branch`, the MATPOWER `rateA = 0` fix, and corrective
generator recourse `pgk = pg0 + extrak` with `0 <= extra <= corrective_action_ratio*pmax`.
Both `form = :polar` and `form = :rect` are supported.

Structure:
- **First stage (design):** the base-case OPF — generation, the form's voltage
  variables, arc flows, and their power-balance / flow / limit constraints.
- **Second stage (`EachScenario`):** for each contingency `k`, the post-contingency
  OPF and its constraints, plus the coupling `pgk - pg0 - extrak = 0` (the only
  rows that touch both stages).

Requires `K = length(contingencies) >= 1`.

# Returns
`(model, vars, cons, post_solve_info)` where
`post_solve_info = (; ns, nv, nd, nc, nc_design, var_scen, con_scen)` gives the Schur
dimensions (scenarios / vars-per-scenario / design vars / cons-per-scenario / design-only
constraints) plus the per-variable and per-constraint scenario tags. Pass the tags through
`kkt_options` so the Schur solver partitions by tag (design and scenario variables are
interleaved here, not contiguous):

```julia
using MadNLP, MadNLPGPU, CUDA, CUDSS
model, vars, cons, info = scopf_twostage_model(case, conts; backend = CUDABackend())
result = madnlp(model;
    kkt_system    = SchurComplementKKTSystem,
    linear_solver = MadNLPGPU.CUDSSSolver,
    kkt_options   = Dict(:schur_ns => info.ns, :schur_nv => info.nv,
                         :schur_nd => info.nd, :schur_nc => info.nc,
                         :schur_var_scen => info.var_scen, :schur_con_scen => info.con_scen),
)
```

!!! note "Design-only base-case constraints"
    The base-case physics are first-stage (design) constraints — `nc_design` of them,
    reported in `post_solve_info`. `SchurComplementKKTSystem` folds the design equalities
    into the bordered first-stage block and condenses the design inequalities into it, so
    this two-stage model and the monolithic `scopf_model` converge to the same optimum.
    The base case produces these rows by construction — they are deliberately NOT worked
    around by replicating base physics into every scenario.
"""
function scopf_twostage_model(
    filename, contingencies;
    backend = nothing,
    T = Float64,
    form = :polar,
    corrective_action_ratio = 0.05,
    unlimited_rate = 1.0e4,
    user_callback = dummy_extension,
    kwargs...,
)
    (form == :polar || form == :rect) || error("scopf_twostage_model supports form = :polar or :rect (got $form).")

    data, K, nbus, narc, ngen, nbranch =
        parse_scopf_twostage_data(filename, contingencies, corrective_action_ratio; unlimited_rate = unlimited_rate, T = T)
    data = convert_data(data, backend)

    # Build the underlying ExaCore explicitly so T is honored (TwoStageExaCore()
    # would force Float64).
    core = ExaCore(T;
        backend = backend,
        tag = ExaModels.TwoStageExaModelTag(
            K,
            convert_array(zeros(Int, 0), backend),
            convert_array(zeros(Int, 0), backend),
        ),
    )

    # ---- Form-independent variables --------------------------------------
    # First-stage (base-case) generation and arc flows.
    @add_var(core, pg0, ngen; lvar = data.pmin, uvar = data.pmax)
    @add_var(core, qg0, ngen; lvar = data.qmin, uvar = data.qmax)
    @add_var(core, p0, narc; lvar = -data.rate_a, uvar = data.rate_a)
    @add_var(core, q0, narc; lvar = -data.rate_a, uvar = data.rate_a)
    # Second-stage (per-contingency) generation, corrective adjustment, and arc flows.
    @add_var(core, pgk, EachScenario(), ngen; lvar = data.pgk_lb, uvar = data.pgk_ub)
    @add_var(core, qgk, EachScenario(), ngen; lvar = data.qgk_lb, uvar = data.qgk_ub)
    @add_var(core, extrak, EachScenario(), ngen; lvar = data.extrak_lb, uvar = data.extrak_ub)
    @add_var(core, pk, EachScenario(), narc; lvar = -data.rate_mat, uvar = data.rate_mat)
    @add_var(core, qk, EachScenario(), narc; lvar = -data.rate_mat, uvar = data.rate_mat)

    # ---- Objective: generation cost averaged over base + K contingencies -
    @add_obj(core, o_base, gen_cost(g, pg0[g.i]) / (K + 1) for g in data.gen)
    @add_obj(core, o_ctg, gen_cost(g, pgk[g.i, k]) / (K + 1) for (g, k) in data.genarray_sc)

    # ---- Corrective coupling (second-stage; links first-stage pg0) -------
    @add_con(core, c_couple, EachScenario(), pgk[g.i, k] - pg0[g.i] - extrak[g.i, k] for (g, k) in data.coupling_sc)

    # ---- Thermal limits (form-independent) -------------------------------
    @add_con(core, c_thermf0, c_thermal_limit(b, p0[b.f_idx], q0[b.f_idx]) for b in data.branch; lcon = fill(-Inf, nbranch))
    @add_con(core, c_thermt0, c_thermal_limit(b, p0[b.t_idx], q0[b.t_idx]) for b in data.branch; lcon = fill(-Inf, nbranch))
    @add_con(core, c_thermfk, EachScenario(), c_thermal_limit(b, pk[b.f_idx, k], qk[b.f_idx, k]) for (b, k) in data.barray_sc; lcon = fill(-Inf, size(data.barray_sc)))
    @add_con(core, c_thermtk, EachScenario(), c_thermal_limit(b, pk[b.t_idx, k], qk[b.t_idx, k]) for (b, k) in data.barray_sc; lcon = fill(-Inf, size(data.barray_sc)))

    if form == :polar
        # ---- Voltage variables ----
        @add_var(core, va0, nbus; lvar = -pi, uvar = pi)
        @add_var(core, vm0, nbus; lvar = data.vmin, uvar = data.vmax, start = 1)
        @add_var(core, vak, EachScenario(), nbus; lvar = -pi, uvar = pi)
        @add_var(core, vmk, EachScenario(), nbus; lvar = data.vmk_lb, uvar = data.vmk_ub, start = 1)

        # ---- First-stage (base-case) constraints ----
        @add_con(core, c_ref0, c_ref_angle_polar(va0[i]) for i in data.ref_buses)
        @add_con(core, c_to_p0, c_to_active_power_flow_polar(b, p0[b.f_idx], vm0[b.f_bus], vm0[b.t_bus], va0[b.f_bus], va0[b.t_bus]) for b in data.branch)
        @add_con(core, c_to_q0, c_to_reactive_power_flow_polar(b, q0[b.f_idx], vm0[b.f_bus], vm0[b.t_bus], va0[b.f_bus], va0[b.t_bus]) for b in data.branch)
        @add_con(core, c_from_p0, c_from_active_power_flow_polar(b, p0[b.t_idx], vm0[b.f_bus], vm0[b.t_bus], va0[b.f_bus], va0[b.t_bus]) for b in data.branch)
        @add_con(core, c_from_q0, c_from_reactive_power_flow_polar(b, q0[b.t_idx], vm0[b.f_bus], vm0[b.t_bus], va0[b.f_bus], va0[b.t_bus]) for b in data.branch)
        @add_con(core, c_phase0, c_phase_angle_diff_polar(b, va0[b.f_bus], va0[b.t_bus]) for b in data.branch; lcon = data.angmin, ucon = data.angmax)
        @add_con(core, c_pbal_p0, c_active_power_balance_demand_polar(b, vm0[b.i]) for b in data.bus)
        @add_con(core, c_pbal_q0, c_reactive_power_balance_demand_polar(b, vm0[b.i]) for b in data.bus)

        # ---- Second-stage (per-contingency) constraints ----
        @add_con(core, c_refk, EachScenario(), c_ref_angle_polar(vak[i, k]) for (i, k) in data.refarray_sc)
        @add_con(core, c_to_pk, EachScenario(), c_to_active_power_flow_polar(b, pk[b.f_idx, k], vmk[b.f_bus, k], vmk[b.t_bus, k], vak[b.f_bus, k], vak[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_to_qk, EachScenario(), c_to_reactive_power_flow_polar(b, qk[b.f_idx, k], vmk[b.f_bus, k], vmk[b.t_bus, k], vak[b.f_bus, k], vak[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_from_pk, EachScenario(), c_from_active_power_flow_polar(b, pk[b.t_idx, k], vmk[b.f_bus, k], vmk[b.t_bus, k], vak[b.f_bus, k], vak[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_from_qk, EachScenario(), c_from_reactive_power_flow_polar(b, qk[b.t_idx, k], vmk[b.f_bus, k], vmk[b.t_bus, k], vak[b.f_bus, k], vak[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_phasek, EachScenario(), c_phase_angle_diff_polar(b, vak[b.f_bus, k], vak[b.t_bus, k]) for (b, k) in data.barray_sc; lcon = data.angmin_sc, ucon = data.angmax_sc)
        @add_con(core, c_pbal_pk, EachScenario(), c_active_power_balance_demand_polar(b, vmk[b.i, k]) for (b, k) in data.busarray_sc)
        @add_con(core, c_pbal_qk, EachScenario(), c_reactive_power_balance_demand_polar(b, vmk[b.i, k]) for (b, k) in data.busarray_sc)

        # ---- Power-balance accumulation (arcs + generation) ----
        @add_con!(core, c_pbal_p0, a.bus => p0[a.i] for a in data.arc)
        @add_con!(core, c_pbal_q0, a.bus => q0[a.i] for a in data.arc)
        @add_con!(core, c_pbal_p0, g.bus => -pg0[g.i] for g in data.gen)
        @add_con!(core, c_pbal_q0, g.bus => -qg0[g.i] for g in data.gen)
        @add_con!(core, c_pbal_pk, a.bus + nbus * (k - 1) => pk[a.i, k] for (a, k) in data.arcarray_sc)
        @add_con!(core, c_pbal_qk, a.bus + nbus * (k - 1) => qk[a.i, k] for (a, k) in data.arcarray_sc)
        @add_con!(core, c_pbal_pk, g.bus + nbus * (k - 1) => -pgk[g.i, k] for (g, k) in data.genarray_sc)
        @add_con!(core, c_pbal_qk, g.bus + nbus * (k - 1) => -qgk[g.i, k] for (g, k) in data.genarray_sc)

        vars = (
            pg0 = pg0, qg0 = qg0, va0 = va0, vm0 = vm0, p0 = p0, q0 = q0,
            pgk = pgk, qgk = qgk, extrak = extrak, vak = vak, vmk = vmk, pk = pk, qk = qk,
        )
        cons = (
            c_ref0 = c_ref0,
            c_to_p0 = c_to_p0, c_to_q0 = c_to_q0, c_from_p0 = c_from_p0, c_from_q0 = c_from_q0,
            c_phase0 = c_phase0, c_pbal_p0 = c_pbal_p0, c_pbal_q0 = c_pbal_q0,
            c_thermf0 = c_thermf0, c_thermt0 = c_thermt0,
            c_couple = c_couple, c_refk = c_refk,
            c_to_pk = c_to_pk, c_to_qk = c_to_qk, c_from_pk = c_from_pk, c_from_qk = c_from_qk,
            c_phasek = c_phasek, c_pbal_pk = c_pbal_pk, c_pbal_qk = c_pbal_qk,
            c_thermfk = c_thermfk, c_thermtk = c_thermtk,
        )

    else  # form == :rect
        # Voltage-magnitude bound squares (vr^2 + vim^2 is bounded directly).
        vmin2 = data.vmin .^ 2; vmax2 = data.vmax .^ 2
        vmink2 = data.vmk_lb .^ 2; vmaxk2 = data.vmk_ub .^ 2

        # ---- Voltage variables ----
        @add_var(core, vr0, nbus; start = 1)
        @add_var(core, vim0, nbus;)
        @add_var(core, vrk, EachScenario(), nbus; start = 1)
        @add_var(core, vimk, EachScenario(), nbus;)

        # ---- First-stage (base-case) constraints ----
        @add_con(core, c_ref0, c_ref_angle_rect(vr0[i], vim0[i]) for i in data.ref_buses)
        @add_con(core, c_to_p0, c_to_active_power_flow_rect(b, p0[b.f_idx], vr0[b.f_bus], vr0[b.t_bus], vim0[b.f_bus], vim0[b.t_bus]) for b in data.branch)
        @add_con(core, c_to_q0, c_to_reactive_power_flow_rect(b, q0[b.f_idx], vr0[b.f_bus], vr0[b.t_bus], vim0[b.f_bus], vim0[b.t_bus]) for b in data.branch)
        @add_con(core, c_from_p0, c_from_active_power_flow_rect(b, p0[b.t_idx], vr0[b.f_bus], vr0[b.t_bus], vim0[b.f_bus], vim0[b.t_bus]) for b in data.branch)
        @add_con(core, c_from_q0, c_from_reactive_power_flow_rect(b, q0[b.t_idx], vr0[b.f_bus], vr0[b.t_bus], vim0[b.f_bus], vim0[b.t_bus]) for b in data.branch)
        @add_con(core, c_phase0, c_phase_angle_diff_rect(b, vr0[b.f_bus], vr0[b.t_bus], vim0[b.f_bus], vim0[b.t_bus]) for b in data.branch; lcon = data.angmin, ucon = data.angmax)
        @add_con(core, c_pbal_p0, c_active_power_balance_demand_rect(b, vr0[b.i], vim0[b.i]) for b in data.bus)
        @add_con(core, c_pbal_q0, c_reactive_power_balance_demand_rect(b, vr0[b.i], vim0[b.i]) for b in data.bus)
        @add_con(core, c_vmag0, c_voltage_magnitude_rect(vr0[b.i], vim0[b.i]) for b in data.bus; lcon = vmin2, ucon = vmax2)

        # ---- Second-stage (per-contingency) constraints ----
        @add_con(core, c_refk, EachScenario(), c_ref_angle_rect(vrk[i, k], vimk[i, k]) for (i, k) in data.refarray_sc)
        @add_con(core, c_to_pk, EachScenario(), c_to_active_power_flow_rect(b, pk[b.f_idx, k], vrk[b.f_bus, k], vrk[b.t_bus, k], vimk[b.f_bus, k], vimk[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_to_qk, EachScenario(), c_to_reactive_power_flow_rect(b, qk[b.f_idx, k], vrk[b.f_bus, k], vrk[b.t_bus, k], vimk[b.f_bus, k], vimk[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_from_pk, EachScenario(), c_from_active_power_flow_rect(b, pk[b.t_idx, k], vrk[b.f_bus, k], vrk[b.t_bus, k], vimk[b.f_bus, k], vimk[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_from_qk, EachScenario(), c_from_reactive_power_flow_rect(b, qk[b.t_idx, k], vrk[b.f_bus, k], vrk[b.t_bus, k], vimk[b.f_bus, k], vimk[b.t_bus, k]) for (b, k) in data.barray_sc)
        @add_con(core, c_phasek, EachScenario(), c_phase_angle_diff_rect(b, vrk[b.f_bus, k], vrk[b.t_bus, k], vimk[b.f_bus, k], vimk[b.t_bus, k]) for (b, k) in data.barray_sc; lcon = data.angmin_sc, ucon = data.angmax_sc)
        @add_con(core, c_pbal_pk, EachScenario(), c_active_power_balance_demand_rect(b, vrk[b.i, k], vimk[b.i, k]) for (b, k) in data.busarray_sc)
        @add_con(core, c_pbal_qk, EachScenario(), c_reactive_power_balance_demand_rect(b, vrk[b.i, k], vimk[b.i, k]) for (b, k) in data.busarray_sc)
        @add_con(core, c_vmagk, EachScenario(), c_voltage_magnitude_rect(vrk[b.i, k], vimk[b.i, k]) for (b, k) in data.busarray_sc; lcon = vmink2, ucon = vmaxk2)

        # ---- Power-balance accumulation (arcs + generation) ----
        @add_con!(core, c_pbal_p0, a.bus => p0[a.i] for a in data.arc)
        @add_con!(core, c_pbal_q0, a.bus => q0[a.i] for a in data.arc)
        @add_con!(core, c_pbal_p0, g.bus => -pg0[g.i] for g in data.gen)
        @add_con!(core, c_pbal_q0, g.bus => -qg0[g.i] for g in data.gen)
        @add_con!(core, c_pbal_pk, a.bus + nbus * (k - 1) => pk[a.i, k] for (a, k) in data.arcarray_sc)
        @add_con!(core, c_pbal_qk, a.bus + nbus * (k - 1) => qk[a.i, k] for (a, k) in data.arcarray_sc)
        @add_con!(core, c_pbal_pk, g.bus + nbus * (k - 1) => -pgk[g.i, k] for (g, k) in data.genarray_sc)
        @add_con!(core, c_pbal_qk, g.bus + nbus * (k - 1) => -qgk[g.i, k] for (g, k) in data.genarray_sc)

        vars = (
            pg0 = pg0, qg0 = qg0, vr0 = vr0, vim0 = vim0, p0 = p0, q0 = q0,
            pgk = pgk, qgk = qgk, extrak = extrak, vrk = vrk, vimk = vimk, pk = pk, qk = qk,
        )
        cons = (
            c_ref0 = c_ref0,
            c_to_p0 = c_to_p0, c_to_q0 = c_to_q0, c_from_p0 = c_from_p0, c_from_q0 = c_from_q0,
            c_phase0 = c_phase0, c_pbal_p0 = c_pbal_p0, c_pbal_q0 = c_pbal_q0, c_vmag0 = c_vmag0,
            c_thermf0 = c_thermf0, c_thermt0 = c_thermt0,
            c_couple = c_couple, c_refk = c_refk,
            c_to_pk = c_to_pk, c_to_qk = c_to_qk, c_from_pk = c_from_pk, c_from_qk = c_from_qk,
            c_phasek = c_phasek, c_pbal_pk = c_pbal_pk, c_pbal_qk = c_pbal_qk, c_vmagk = c_vmagk,
            c_thermfk = c_thermfk, c_thermtk = c_thermtk,
        )
    end

    # Schur dimensions AND the full per-variable/per-constraint scenario tags.
    # MadNLP's SchurComplementKKTSystem partitions by these tags (design and
    # scenario variables are interleaved here, not contiguous), and folds the
    # design-only base-case constraints into its first-stage bordered block.
    # Pass them through `kkt_options` as :schur_var_scen / :schur_con_scen.
    var_scen = Vector{Int}(Array(core.tag.var_scen))
    con_scen = Vector{Int}(Array(core.tag.con_scen))
    post_solve_info = (
        ns = K,
        nv = count(==(1), var_scen),
        nd = count(==(0), var_scen),
        nc = count(==(1), con_scen),
        nc_design = count(==(0), con_scen),
        var_scen = var_scen,
        con_scen = con_scen,
    )

    vars2, cons2 = user_callback(core, vars, cons)
    model = ExaModel(core; kwargs...)

    vars = (; vars..., vars2...)
    cons = (; cons..., cons2...)
    return model, vars, cons, post_solve_info
end
