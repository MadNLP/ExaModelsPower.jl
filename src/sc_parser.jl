is_pr(uid::Int, L_J_pr::Int, L_J_cs::Int, producers_first::Bool)::Bool = producers_first ? uid < L_J_pr : uid >= L_J_cs
function is_pr(uid_str::String, L_J_pr::Int, L_J_cs::Int, producers_first::Bool)::Bool
    uid = parse(Int, match(r"\d+", uid_str).match)
    is_pr(uid, L_J_pr, L_J_cs, producers_first)
end

function get_j_prcs(uid_str::String, L_J_pr::Int, L_J_cs::Int, producers_first::Bool)
    parse(Int, match(r"\d+", uid_str).match) + 1
end

function get_j_pr(uid_str::String, L_J_pr::Int, L_J_cs::Int, producers_first::Bool)
    offset = Int64(producers_first ? 0 : (-L_J_cs))
    parse(Int, match(r"\d+", uid_str).match) + 1 + offset
end

function get_j_cs(uid_str::String, L_J_pr::Int, L_J_cs::Int, producers_first::Bool)
    offset = Int64(producers_first ? (-L_J_pr) : 0)
    parse(Int, match(r"\d+", uid_str).match) + 1 + offset
end

function parse_sc_data_static(data, producers_first)
    L_J_xf = length(data.twt_lookup)
    L_J_ln = length(data.ac_line_lookup)
    L_J_ac = L_J_ln + L_J_xf
    L_J_dc = length(data.dc_line_lookup)
    L_J_br =  L_J_ac + L_J_dc
    L_J_cs = length(data.sdd_ids_consumer)
    L_J_pr = length(data.sdd_ids_producer)
    L_J_cspr = L_J_cs + L_J_pr
    L_J_sh = length(data.shunt_lookup)
    L_N_p = length(data.azr_lookup)
    L_N_q = length(data.rzr_lookup)
    I  = length(data.bus_lookup)
    L_T = length(data.dt)

    lengths = (L_J_xf=L_J_xf, L_J_ln=L_J_ln, L_J_ac=L_J_ac, L_J_dc=L_J_dc, L_J_br=L_J_br, L_J_cs=L_J_cs,
    L_J_pr=L_J_pr, L_J_cspr = L_J_cspr, L_J_sh=L_J_sh, I=I, L_T=L_T, L_N_p, L_N_q)

    ε_time = 1e-6

    cost_vector_pr = sort(
            #producers
            [
                begin
                    ts_val = data.sdd_ts_lookup[key]
                    j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1
                    j_prcs = get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    j_pr = get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
                    bus = parse(Int, match(r"\d+", val["bus"]).match) + 1
                    uid = val["uid"]
                    cost = ts_val["cost"]
                    (j = j, bus=bus, uid = uid, cost=cost, j_pr=j_pr, j_prcs=j_prcs)
                end for (key, val) in data.sdd_lookup if val["device_type"] == "producer"
                
            ], by = x -> x.j)

            #Consumers
    cost_vector_cs = sort(
            [
                begin
                    ts_val = data.sdd_ts_lookup[key]
                    j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1
                    j_prcs = get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    j_cs = get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    bus = parse(Int, match(r"\d+", val["bus"]).match) + 1
                    uid = val["uid"]
                    cost = ts_val["cost"]
                    (j = j, bus=bus, uid = uid, cost=cost, j_cs=j_cs, j_prcs=j_prcs)
                end for (key, val) in data.sdd_lookup if val["device_type"] == "consumer"
            ], by = x -> x.j)

    
    sc_data = (
        bus = sort([
            begin
                i = parse(Int, match(r"\d+", val["uid"]).match)+1
                uid = val["uid"]
                v_min = val["vm_lb"]
                v_max = val["vm_ub"]
                (i = i, uid = uid, v_min = v_min, v_max = v_max)
            end for val in values(data.bus_lookup)
        ], by = x -> x.i),

        shunt = sort([
            begin
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + L_J_cspr+1
                j_sh = parse(Int, match(r"\d+", val["uid"]).match)+1
                uid = val["uid"]
                bus = parse(Int, match(r"\d+", val["bus"]).match)+1
                g_sh = val["gs"]
                b_sh = val["bs"]
                (j = j, j_sh=j_sh, uid = uid, bus=bus, g_sh = g_sh, b_sh = b_sh)
            end for val in values(data.shunt_lookup)
        ], by = x -> x.j),

        acl_branch = sort(
            # AC lines
            [
                begin
                    j = parse(Int, match(r"\d+", val["uid"]).match)+1
                    j_ac = j
                    j_ln = j
                    uid = val["uid"]
                    to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1
                    fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1
                    c_su = Float64(val["connection_cost"])
                    c_sd = Float64(val["disconnection_cost"])
                    s_max = Float64(val["mva_ub_nom"])
                    r = Float64(val["r"])
                    x = Float64(val["x"])
                    g_sr = Float64(r / (x^2 + r^2))
                    b_sr = Float64(-x / (x^2 + r^2))
                    b_ch = Float64(val["b"])
                    if val["additional_shunt"] == 1
                        g_fr = Float64(val["g_fr"])
                        g_to = Float64(val["g_to"])
                        b_fr = Float64(val["b_fr"])
                        b_to = Float64(val["b_to"])
                    else
                        g_fr = 0.0
                        g_to = 0.0
                        b_fr = 0.0
                        b_to = 0.0
                    end
                    (j = j, j_ac = j_ac, j_ln = j_ln, uid = uid, to_bus = to_bus, fr_bus = fr_bus, c_su = c_su, c_sd = c_sd, s_max = s_max, 
                    g_sr = g_sr, b_sr = b_sr, b_ch = b_ch, g_fr = g_fr, g_to = g_to, b_fr = b_fr, b_to = b_to)
                end for val in values(data.ac_line_lookup)
            ],by = x -> x.j),
            
            # Transformers
        acx_branch = sort(
            [
                begin
                    j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1
                    j = j_xf + L_J_ln
                    j_ac = j
                    uid = val["uid"]
                    to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1
                    fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1
                    c_su = val["connection_cost"]
                    c_sd = val["disconnection_cost"]
                    s_max = val["mva_ub_nom"]
                    r = val["r"]
                    x = val["x"]
                    g_sr = r / (x^2 + r^2)
                    b_sr = -x / (x^2 + r^2)
                    b_ch = val["b"]
                    if val["additional_shunt"] == 1
                        g_fr = val["g_fr"]
                        g_to = val["g_to"]
                        b_fr = val["b_fr"]
                        b_to = val["b_to"]
                    else
                        g_fr = 0
                        g_to = 0
                        b_fr = 0
                        b_to = 0
                    end
                    (j = j, j_ac = j_ac, j_xf=j_xf, uid = uid, to_bus = to_bus, fr_bus = fr_bus, c_su = c_su, c_sd = c_sd, s_max = s_max, 
                    g_sr = g_sr, b_sr = b_sr, b_ch = b_ch, g_fr = g_fr, g_to = g_to, b_fr = b_fr, b_to = b_to)                
                end for val in values(data.twt_lookup)
            ]
        , by = x -> x.j),
        #Variable phase difference
        vpd = isempty(val for val in values(data.twt_lookup) if val["ta_lb"] < val["ta_ub"]) ? empty_data = Vector{NamedTuple{(:j,), Tuple{Int64}}}() : sort([
            begin
                j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_ln+1
                j_ac = j
                phi_min = val["ta_lb"]
                phi_max = val["ta_ub"]
                (j = j, j_ac = j_ac, j_xf=j_xf, phi_min = phi_min, phi_max = phi_max)
            end for val in values(data.twt_lookup) if val["ta_lb"] < val["ta_ub"]
        ], by = x -> x.j),
        #Fixed phase difference
        fpd = isempty(val for val in values(data.twt_lookup) if val["ta_lb"] >= val["ta_ub"]) ? empty_data = Vector{NamedTuple{(:j,), Tuple{Int64}}}() : sort([
            begin
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_ln+1
                j_ac = j
                j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1
                phi_o = val["initial_status"]["ta"]
                (j = j, j_ac = j_ac, j_xf=j_xf, phi_o = phi_o)
            end for val in values(data.twt_lookup) if val["ta_lb"] >= val["ta_ub"]
        ], by = x -> x.j),
        #Variable winding ratio
        vwr = isempty(val for val in values(data.twt_lookup) if val["tm_lb"] < val["tm_ub"]) ? empty_data = Vector{NamedTuple{(:j,), Tuple{Int64}}}() : sort([
            begin
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_ln+1
                j_ac = j
                j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1
                tau_min = val["tm_lb"]
                tau_max = val["tm_ub"]
                (j=j, j_ac=j_ac, j_xf=j_xf, tau_min=tau_min, tau_max=tau_max)
            end for val in values(data.twt_lookup) if val["tm_lb"] < val["tm_ub"]
        ], by = x -> x.j),
        #Fixed winding ratio
        fwr = isempty(val for val in values(data.twt_lookup) if val["tm_lb"] >= val["tm_ub"]) ? empty_data = Vector{NamedTuple{(:j,), Tuple{Int64}}}() : sort([
            begin
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_ln+1
                j_ac = j
                j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1
                tau_o = val["initial_status"]["tm"]
                (j=j, j_ac=j_ac, j_xf=j_xf, tau_o=tau_o)
            end for val in values(data.twt_lookup) if val["tm_lb"] >= val["tm_ub"]

        ], by = x -> x.j),

        dc_branch = L_J_dc == 0 ? empty_data = Vector{@NamedTuple{j::Int, j_dc::Int, uid::String,pdc_max::Float64, qdc_fr_min::Float64, qdc_to_min::Float64, qdc_fr_max::Float64, qdc_to_max::Float64, to_bus::Int, fr_bus::Int}}() :  sort([
            begin
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_ac+1
                j_dc = parse(Int, match(r"\d+", val["uid"]).match)+1
                uid = val["uid"]
                pdc_max = val["pdc_ub"]
                qdc_fr_min = val["qdc_fr_lb"]
                qdc_to_min = val["qdc_to_lb"]
                qdc_fr_max = val["qdc_fr_ub"]
                qdc_to_max = val["qdc_to_ub"]
                to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1
                fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1
                (j=j, j_dc = j_dc, uid=uid, pdc_max=pdc_max, qdc_fr_min=qdc_fr_min, qdc_to_min=qdc_to_min, qdc_fr_max=qdc_fr_max, qdc_to_max=qdc_to_max, to_bus=to_bus, fr_bus=fr_bus)
            end for val in values(data.dc_line_lookup)

        ], by = x -> x.j),
 
        
        prod = sort(
            # Producers
            [
                begin
                    ts_val = data.sdd_ts_lookup[key]
                    j = Int(parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1)
                    j_pr = get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
                    j_prcs = get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    bus = Int(parse(Int, match(r"\d+", val["bus"]).match) + 1)
                    uid = String(val["uid"])
                    c_on = Float64(val["on_cost"])
                    c_su = Float64(val["startup_cost"])
                    c_sd = Float64(val["shutdown_cost"])
                    p_ru = Float64(val["p_ramp_up_ub"])
                    p_rd = Float64(val["p_ramp_down_ub"])
                    p_ru_su = Float64(val["p_startup_ramp_ub"])
                    p_rd_sd = Float64(val["p_shutdown_ramp_ub"])
                    c_rgu = convert(Vector{Float64}, ts_val["p_reg_res_up_cost"]) #these need checks to see if empty
                    c_rgd = convert(Vector{Float64}, ts_val["p_reg_res_down_cost"])
                    c_scr = convert(Vector{Float64}, ts_val["p_syn_res_cost"])
                    c_nsc = convert(Vector{Float64}, ts_val["p_nsyn_res_cost"])
                    c_rru_on = convert(Vector{Float64}, ts_val["p_ramp_res_up_online_cost"])
                    c_rru_off = convert(Vector{Float64}, ts_val["p_ramp_res_up_offline_cost"])
                    c_rrd_on = convert(Vector{Float64}, ts_val["p_ramp_res_down_online_cost"])
                    c_rrd_off = convert(Vector{Float64}, ts_val["p_ramp_res_down_offline_cost"])
                    c_qru = convert(Vector{Float64}, ts_val["q_res_up_cost"])
                    c_qrd = convert(Vector{Float64}, ts_val["q_res_down_cost"])
                    p_rgu_max = Float64(val["p_reg_res_up_ub"])
                    p_rgd_max = Float64(val["p_reg_res_down_ub"])
                    p_scr_max = Float64(val["p_syn_res_ub"])
                    p_nsc_max = Float64(val["p_nsyn_res_ub"])
                    p_rru_on_max = Float64(val["p_ramp_res_up_online_ub"])
                    p_rru_off_max = Float64(val["p_ramp_res_up_offline_ub"])
                    p_rrd_on_max = Float64(val["p_ramp_res_down_online_ub"])
                    p_rrd_off_max = Float64(val["p_ramp_res_down_offline_ub"])
                    p_0 = Float64(val["initial_status"]["p"])
                    q_0 = Float64(val["initial_status"]["q"])
                    p_max = convert(Vector{Float64}, ts_val["p_ub"])
                    p_min = convert(Vector{Float64}, ts_val["p_lb"])
                    q_max = convert(Vector{Float64}, ts_val["q_ub"])
                    q_min = convert(Vector{Float64}, ts_val["q_lb"])
                    sus = convert(Vector{Vector{Float64}}, val["startup_states"])
                    
                    (j = j, j_pr=j_pr, j_prcs=j_prcs, bus=bus, uid = uid, c_on = c_on, c_su = c_su, c_sd = c_sd, p_ru = p_ru, p_rd = p_rd, p_ru_su = p_ru_su, p_rd_sd = p_rd_sd, 
                    c_rgu = c_rgu, c_rgd = c_rgd, c_scr = c_scr, c_nsc = c_nsc, c_rru_on = c_rru_on, c_rru_off = c_rru_off, c_rrd_on = c_rrd_on, c_rrd_off = c_rrd_off, 
                    c_qru = c_qru, c_qrd = c_qrd, p_rgu_max = p_rgu_max, p_rgd_max = p_rgd_max, p_scr_max = p_scr_max, p_nsc_max = p_nsc_max, p_rru_on_max = p_rru_on_max,
                    p_rru_off_max=p_rru_off_max, p_rrd_on_max=p_rrd_on_max, p_rrd_off_max=p_rrd_off_max, p_0=p_0, q_0=q_0, p_max=p_max, p_min=p_min, q_max=q_max, q_min=q_min, sus=sus)
                end for (key, val) in data.sdd_lookup if val["device_type"] == "producer"
            ], by = x -> x.j),
            
        #Consumers
        cons = sort(
            [
                begin
                    ts_val = data.sdd_ts_lookup[key]
                    j = Int(parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1)
                    j_prcs = get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    j_cs = get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first)
                    bus = Int(parse(Int, match(r"\d+", val["bus"]).match) + 1)
                    uid = String(val["uid"])
                    c_on = Float64(val["on_cost"])
                    c_su = Float64(val["startup_cost"])
                    c_sd = Float64(val["shutdown_cost"])
                    p_ru = Float64(val["p_ramp_up_ub"])
                    p_rd = Float64(val["p_ramp_down_ub"])
                    p_ru_su = Float64(val["p_startup_ramp_ub"])
                    p_rd_sd = Float64(val["p_shutdown_ramp_ub"])
                    c_rgu = convert(Vector{Float64}, ts_val["p_reg_res_up_cost"])
                    c_rgd = convert(Vector{Float64}, ts_val["p_reg_res_down_cost"])
                    c_scr = convert(Vector{Float64}, ts_val["p_syn_res_cost"])
                    c_nsc = convert(Vector{Float64}, ts_val["p_nsyn_res_cost"])
                    c_rru_on = convert(Vector{Float64}, ts_val["p_ramp_res_up_online_cost"])
                    c_rru_off = convert(Vector{Float64}, ts_val["p_ramp_res_up_offline_cost"])
                    c_rrd_on = convert(Vector{Float64}, ts_val["p_ramp_res_down_online_cost"])
                    c_rrd_off = convert(Vector{Float64}, ts_val["p_ramp_res_down_offline_cost"])
                    c_qru = convert(Vector{Float64}, ts_val["q_res_up_cost"])
                    c_qrd = convert(Vector{Float64}, ts_val["q_res_down_cost"])
                    p_rgu_max = Float64(val["p_reg_res_up_ub"])
                    p_rgd_max = Float64(val["p_reg_res_down_ub"])
                    p_scr_max = Float64(val["p_syn_res_ub"])
                    p_nsc_max = Float64(val["p_nsyn_res_ub"])
                    p_rru_on_max = Float64(val["p_ramp_res_up_online_ub"])
                    p_rru_off_max = Float64(val["p_ramp_res_up_offline_ub"])
                    p_rrd_on_max = Float64(val["p_ramp_res_down_online_ub"])
                    p_rrd_off_max = Float64(val["p_ramp_res_down_offline_ub"])
                    p_0 = Float64(val["initial_status"]["p"])
                    q_0 = Float64(val["initial_status"]["q"])
                    p_max = convert(Vector{Float64}, ts_val["p_ub"])
                    p_min = convert(Vector{Float64}, ts_val["p_lb"])
                    q_max = convert(Vector{Float64}, ts_val["q_ub"])
                    q_min = convert(Vector{Float64}, ts_val["q_lb"])
                    sus = convert(Vector{Vector{Float64}}, val["startup_states"])

                    (j = j, j_cs=j_cs, j_prcs=j_prcs, bus=bus, uid = uid, c_on = c_on, c_su = c_su, c_sd = c_sd, p_ru = p_ru, p_rd = p_rd, p_ru_su = p_ru_su, p_rd_sd = p_rd_sd, 
                    c_rgu = c_rgu, c_rgd = c_rgd, c_scr = c_scr, c_nsc = c_nsc, c_rru_on = c_rru_on, c_rru_off = c_rru_off, c_rrd_on = c_rrd_on, c_rrd_off = c_rrd_off, 
                    c_qru = c_qru, c_qrd = c_qrd, p_rgu_max = p_rgu_max, p_rgd_max = p_rgd_max, p_scr_max = p_scr_max, p_nsc_max = p_nsc_max, p_rru_on_max = p_rru_on_max,
                    p_rru_off_max=p_rru_off_max, p_rrd_on_max=p_rrd_on_max, p_rrd_off_max=p_rrd_off_max, p_0=p_0, q_0=q_0, p_max=p_max, p_min=p_min, q_max=q_max, q_min=q_min, sus=sus)
                end for (key, val) in data.sdd_lookup if val["device_type"] == "consumer"
            ]
        , by = x -> x.j),
        active_reserve = sort([
            begin
                ts_val = data.azr_ts_lookup[key]
                n = parse(Int, match(r"\d+", val["uid"]).match) + 1
                n_p = n
                uid = val["uid"]
                c_rgu = val["REG_UP_vio_cost"]
                c_rgd = val["REG_DOWN_vio_cost"]
                c_scr = val["SYN_vio_cost"]
                c_nsc = val["NSYN_vio_cost"]
                c_rru = val["RAMPING_RESERVE_UP_vio_cost"]
                c_rrd = val["RAMPING_RESERVE_DOWN_vio_cost"]
                σ_rgu = val["REG_UP"]
                σ_rgd = val["REG_DOWN"]
                σ_scr = val["SYN"]
                σ_nsc = val["NSYN"]
                p_rru_min = convert(Vector{Float64}, ts_val["RAMPING_RESERVE_UP"])
                p_rrd_min = convert(Vector{Float64}, ts_val["RAMPING_RESERVE_DOWN"])
                (n=n, n_p=n_p, uid=uid, c_rgu=c_rgu, c_rgd=c_rgd, c_scr=c_scr, c_nsc=c_nsc, c_rru=c_rru, c_rrd=c_rrd, σ_rgu=σ_rgu, σ_rgd=σ_rgd, σ_scr=σ_scr, 
                σ_nsc=σ_nsc, p_rru_min=p_rru_min, p_rrd_min=p_rrd_min)
            end for (key, val) in data.azr_lookup
        ], by = x -> x.n),
        reactive_reserve = sort([
            begin
                ts_val = data.rzr_ts_lookup[key]
                n = parse(Int, match(r"\d+", val["uid"]).match) + L_N_p + 1
                n_q = parse(Int, match(r"\d+", val["uid"]).match) + 1
                uid = val["uid"]
                c_qru = val["REACT_UP_vio_cost"]
                c_qrd = val["REACT_DOWN_vio_cost"]
                q_qru_min = convert(Vector{Float64}, ts_val["REACT_UP"])
                q_qrd_min = convert(Vector{Float64}, ts_val["REACT_DOWN"])
                (n=n, n_q=n_q, uid=uid, c_qru=c_qru, c_qrd=c_qrd, q_qru_min=q_qru_min, q_qrd_min=q_qrd_min)
            end for (key, val) in data.rzr_lookup
        ], by = x -> x.n),

        active_reserve_set_pr = [
            (i = parse(Int, match(r"\d+", bus["uid"]).match) + 1,
             j = parse(Int, match(r"\d+", device["uid"]).match) + L_J_br + 1,
             n = parse(Int, match(r"\d+", uid).match) + 1,
             n_p =  parse(Int, match(r"\d+", uid).match) + 1,
             j_pr=get_j_pr(device["uid"], L_J_pr, L_J_cs, producers_first),
             j_prcs=get_j_prcs(device["uid"], L_J_pr, L_J_cs, producers_first),
            )
            for uid in data.azr_ids
            for bus in values(data.bus_lookup)
            if uid in bus["active_reserve_uids"]
            for device in values(data.sdd_lookup)
            if device["bus"] == bus["uid"] && is_pr(device["uid"], L_J_pr, L_J_cs, producers_first)
        ],

        active_reserve_set_cs = [
            (i = parse(Int, match(r"\d+", bus["uid"]).match) + 1,
             j = parse(Int, match(r"\d+", device["uid"]).match) + L_J_br + 1,
             n = parse(Int, match(r"\d+", uid).match) + 1,
             n_p =  parse(Int, match(r"\d+", uid).match) + 1,
             j_cs=get_j_cs(device["uid"], L_J_pr, L_J_cs, producers_first),
             j_prcs=get_j_prcs(device["uid"], L_J_pr, L_J_cs, producers_first),)
            for uid in data.azr_ids
            for bus in values(data.bus_lookup)
            if uid in bus["active_reserve_uids"]
            for device in values(data.sdd_lookup)
            if device["bus"] == bus["uid"] && !is_pr(device["uid"], L_J_pr, L_J_cs, producers_first)
        ],

        reactive_reserve_set_pr = [
            (i = parse(Int, match(r"\d+", bus["uid"]).match) + 1,
             j = parse(Int, match(r"\d+", device["uid"]).match) + L_J_br + 1,
             n = parse(Int, match(r"\d+", uid).match) + L_N_p + 1,
             n_q = parse(Int, match(r"\d+", uid).match) + 1,
             j_pr=get_j_pr(device["uid"], L_J_pr, L_J_cs, producers_first),
             j_prcs=get_j_prcs(device["uid"], L_J_pr, L_J_cs, producers_first),)
            for uid in data.rzr_ids
            for bus in values(data.bus_lookup)
            if uid in bus["reactive_reserve_uids"]
            for device in values(data.sdd_lookup)
            if device["bus"] == bus["uid"] && is_pr(device["uid"], L_J_pr, L_J_cs, producers_first)
        ],

        reactive_reserve_set_cs = [
            (i = parse(Int, match(r"\d+", bus["uid"]).match) + 1,
             j = parse(Int, match(r"\d+", device["uid"]).match) + L_J_br + 1,
             n = parse(Int, match(r"\d+", uid).match) + L_N_p + 1,
             n_q = parse(Int, match(r"\d+", uid).match) + 1,
             j_cs=get_j_cs(device["uid"], L_J_pr, L_J_cs, producers_first),
             j_prcs=get_j_prcs(device["uid"], L_J_pr, L_J_cs, producers_first),)
            for uid in data.rzr_ids
            for bus in values(data.bus_lookup)
            if uid in bus["reactive_reserve_uids"]
            for device in values(data.sdd_lookup)
            if device["bus"] == bus["uid"] && !is_pr(device["uid"], L_J_pr, L_J_cs, producers_first)
        ],
  
    )
    return sc_data, lengths, cost_vector_pr, cost_vector_cs
end

function add_status_flags(uc_data, data)
    for dict in uc_data
        uid = dict["uid"]
        on_status = dict["on_status"]
        n = length(on_status)
        u_on_o = data[uid]["initial_status"]["on_status"]

        su_status = zeros(Int, n)
        sd_status = zeros(Int, n)

        # Handle first index using u_on_o if provided
        
        if u_on_o == 0 && on_status[1] == 1
            su_status[1] = 1
        elseif u_on_o == 1 && on_status[1] == 0
            sd_status[1] = 1
        end
        

        # Iterate from 1 to n-1 for the rest
        for i in 1:n-1
            if on_status[i] == 0 && on_status[i+1] == 1
                su_status[i+1] = 1
            end
            if on_status[i] == 1 && on_status[i+1] == 0
                sd_status[i+1] = 1
            end
        end

        dict["su_status"] = su_status
        dict["sd_status"] = sd_status
    end
end

function get_as(dt, t)
    a_end = sum(dt[1:t])
    a_start = a_end - dt[t]
    a_mid = (a_start + a_end)/2
    return a_start, a_mid, a_end
end

function parse_sc_data(data, uc_data, data_json)
    producers_first = data.sdd_lookup[minimum(keys(data.sdd_lookup))]["device_type"] == "producer"
    sc_data, lengths, cost_vector_pr, cost_vector_cs = parse_sc_data_static(data, producers_first)
    @info typeof(sc_data)
    
    (L_J_xf, L_J_ln, L_J_ac, L_J_dc, L_J_br, L_J_cs,
    L_J_pr, L_J_cspr, L_J_sh, I, L_T, L_N_p, L_N_q) = lengths
    periods = data.periods
    dt = Float64.(data.dt)
    K = length(data_json["reliability"]["contingency"])
    add_status_flags(uc_data["time_series_output"]["ac_line"], data.ac_line_lookup)
    add_status_flags(uc_data["time_series_output"]["two_winding_transformer"], data.twt_lookup)
    add_status_flags(uc_data["time_series_output"]["simple_dispatchable_device"], data.sdd_lookup)
    ε_time = 1e-6

    T_supc_pr = [
            (j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
            j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
            t = t,
            t_prime = t_prime, 
            p_supc = data.sdd_ts_lookup[key]["p_lb"][t_prime] - val["p_startup_ramp_ub"]*(get_as(dt, t_prime)[3] - get_as(dt, t)[3]),
            u_su = uc["su_status"][t_prime]
        )
        for (key, val) in data.sdd_lookup
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
        for t in periods
        for t_prime in periods
        if t_prime > t && data.sdd_ts_lookup[key]["p_lb"][t_prime] - val["p_startup_ramp_ub"]*(get_as(dt, t_prime)[3] - get_as(dt, t)[3]) > 0
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if val["uid"] == uc["uid"]
        ]
    
    #This sum corresponds to constraint 69 (summing p_supc*u_su)
    sum_T_supc_pr = zeros(L_J_pr, length(periods))
    #This sum corresponds to constraint 112/113 (summing u_su)
    sum2_T_supc_pr = zeros(L_J_pr, length(periods))

    for b in T_supc_pr
        sum_T_supc_pr[b.j_pr, b.t] += b.p_supc*b.u_su
        sum2_T_supc_pr[b.j_pr, b.t] += b.u_su
    end

    T_supc_cs = [
            (j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
            j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
            t = t,
            t_prime = t_prime, 
            p_supc = data.sdd_ts_lookup[key]["p_lb"][t_prime] - val["p_startup_ramp_ub"]*(get_as(dt, t_prime)[3] - get_as(dt, t)[3]),
            u_su = uc["su_status"][t_prime]
        )
        for (key, val) in data.sdd_lookup
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
        for t in periods
        for t_prime in periods
        if t_prime > t && data.sdd_ts_lookup[key]["p_lb"][t_prime] - val["p_startup_ramp_ub"]*(get_as(dt, t_prime)[3] - get_as(dt, t)[3]) > 0
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if val["uid"] == uc["uid"]
        ]
    #This sum corresponds to constraint 69 (p_supc*u_su)
    sum_T_supc_cs = zeros(L_J_cs, L_T)
    #This sum corresponds to constraints 122-126 (u_su)
    sum2_T_supc_cs = zeros(L_J_cs, L_T)
    for b in T_supc_cs
        sum_T_supc_cs[b.j_cs, b.t] += b.p_supc*b.u_su
        sum2_T_supc_cs[b.j_cs, b.t] += b.u_su
    end


    #Build a p_sdpc set to be used for T_sdpc
    #Index order is j_prcs, t, t_prime
    p_sdpc = zeros(L_J_cspr, L_T, L_T)

    for t in periods
        for t_prime in periods
            for (key, val) in data.sdd_lookup
                if t_prime == 1 && t >= t_prime
                    p_sdpc[get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first), t, t_prime] = val["initial_status"]["p"] - val["p_shutdown_ramp_ub"]*(get_as(dt, t)[3] - get_as(dt, t_prime)[1])
                elseif t >= t_prime
                    p_sdpc[get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first), t, t_prime] = data.sdd_ts_lookup[key]["p_lb"][t_prime-1]- val["p_shutdown_ramp_ub"]*(get_as(dt, t)[3] - get_as(dt, t_prime)[1])
                end
            end
        end
    end

    T_sdpc_pr = [
            (j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
            j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
            t = t,
            t_prime = t_prime,
            p_sdpc = p_sdpc[parse(Int, match(r"\d+", val["uid"]).match)+1, t, t_prime],
            u_sd = uc["sd_status"][t_prime]
            )
        for (key, val) in data.sdd_lookup
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
        for t in periods
        for t_prime in periods
        if t_prime <= t && p_sdpc[parse(Int, match(r"\d+", val["uid"]).match)+1, t, t_prime] > 0
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if val["uid"] == uc["uid"]
        ]

    #This sum corresponds to constraint 70 (summing p_sdpc*u_sd)
    sum_T_sdpc_pr = zeros(L_J_pr, L_T)
    #This sum corresponds to constraints 112 and 113 (summing u_sd)
    sum2_T_sdpc_pr = zeros(L_J_pr, L_T)
    for b in T_sdpc_pr
        sum_T_sdpc_pr[b.j_pr, b.t] += b.p_sdpc*b.u_sd
        sum2_T_sdpc_pr[b.j_pr, b.t] += b.u_sd
    end

    T_sdpc_cs = [
            (j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
            j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
            t = t,
            t_prime = t_prime,
            p_sdpc = p_sdpc[parse(Int, match(r"\d+", val["uid"]).match)+1, t, t_prime],
            u_sd = uc["sd_status"][t_prime]
            )
        for (key, val) in data.sdd_lookup
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
        for t in periods
        for t_prime in periods
        if t_prime <= t && p_sdpc[parse(Int, match(r"\d+", val["uid"]).match)+1, t, t_prime] > 0
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if val["uid"] == uc["uid"]
        ]
    #This sum corresponds to constraint 70 (p_sdpc*u_sd)
    sum_T_sdpc_cs = zeros(L_J_cs, L_T)
    #This sum corresponds to constraint 122-126 (u_sd)
    sum2_T_sdpc_cs = zeros(L_J_cs, L_T)
    for b in T_sdpc_cs
        sum_T_sdpc_cs[b.j_cs, b.t] += b.p_sdpc*b.u_sd
        sum2_T_sdpc_cs[b.j_cs, b.t] += b.u_sd
    end
    

    W_en_max_pr = Vector{@NamedTuple{w_en_max_pr_ind::Int, j::Int, j_prcs::Int, j_pr::Int, a_en_max_start::Float64, a_en_max_end::Float64, e_max::Float64}}()
    w_en_max_pr_ind = 1
    for val in values(data.sdd_lookup) 
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_ub"]
                push!(W_en_max_pr, (
                    w_en_max_pr_ind=w_en_max_pr_ind,
                    j=parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                    j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                    j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                    a_en_max_start = w[1],
                    a_en_max_end = w[2],
                    e_max = w[3])
                )
                w_en_max_pr_ind +=1
            end
        end
    end

    L_W_en_max_pr = length(W_en_max_pr)

    W_en_max_cs = Vector{@NamedTuple{w_en_max_cs_ind::Int, j::Int, j_prcs::Int, j_cs::Int, a_en_max_start::Float64, a_en_max_end::Float64, e_max::Float64}}()
    w_en_max_cs_ind = 1
    for val in values(data.sdd_lookup) 
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_ub"]
                push!(W_en_max_cs, (
                    w_en_max_cs_ind=w_en_max_cs_ind,
                    j=parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                    j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                    j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                    a_en_max_start = w[1],
                    a_en_max_end = w[2],
                    e_max = w[3])
                )
                w_en_max_cs_ind +=1
            end
        end
    end

    L_W_en_max_cs = length(W_en_max_cs)

    W_en_min_pr = Vector{@NamedTuple{w_en_min_pr_ind::Int, j::Int, j_prcs::Int, j_pr::Int, a_en_min_start::Float64, a_en_min_end::Float64, e_min::Float64}}()
    w_en_min_pr_ind = 1
    for val in values(data.sdd_lookup)
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_lb"]
                push!(W_en_min_pr, (w_en_min_pr_ind = w_en_min_pr_ind,
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                a_en_min_start = w[1],
                a_en_min_end = w[2],
                e_min = w[3]))
                w_en_min_pr_ind += 1
            end
        end
    end

    L_W_en_min_pr = length(W_en_min_pr)

    W_en_min_cs = Vector{@NamedTuple{w_en_min_cs_ind::Int, j::Int, j_prcs::Int, j_cs::Int, a_en_min_start::Float64, a_en_min_end::Float64, e_min::Float64}}()
    w_en_min_cs_ind = 1
    for val in values(data.sdd_lookup)
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_lb"]
                push!(W_en_min_cs, (w_en_min_cs_ind = w_en_min_cs_ind,
                j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                a_en_min_start = w[1],
                a_en_min_end = w[2],
                e_min = w[3]))
                w_en_min_cs_ind += 1
            end
        end
    end

    L_W_en_min_cs = length(W_en_min_cs)

    T_w_en_max_pr = Vector{@NamedTuple{w_en_max_pr_ind::Int, j::Int, j_prcs::Int, j_pr::Int, t::Int, dt::Float64}}()
    w_en_max_pr_ind = 0
    for val in values(data.sdd_lookup)
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_ub"]
                w_en_max_pr_ind += 1
                for t in periods
                    if w[1] + ε_time < get_as(dt, t)[2] && get_as(dt, t)[2] <= w[2] + ε_time
                        push!(T_w_en_max_pr, (w_en_max_pr_ind = w_en_max_pr_ind,
                            j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                            t = t,
                            dt=dt[t]))
                    end
                end
            end
        end
    end

    T_w_en_max_cs = Vector{@NamedTuple{w_en_max_cs_ind::Int, j::Int, j_prcs::Int, j_cs::Int, t::Int, dt::Float64}}()
    w_en_max_cs_ind = 0
    for val in values(data.sdd_lookup)
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_ub"]
                w_en_max_cs_ind += 1
                for t in periods
                    if w[1] + ε_time < get_as(dt, t)[2] && get_as(dt, t)[2] <= w[2] + ε_time
                        push!(T_w_en_max_cs, (w_en_max_cs_ind = w_en_max_cs_ind,
                            j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            t = t,
                            dt=dt[t]))
                    end
                end
            end
        end
    end

    T_w_en_min_pr = Vector{@NamedTuple{w_en_min_pr_ind::Int, j::Int, j_prcs::Int, j_pr::Int, t::Int, dt::Float64}}()
    w_en_min_pr_ind = 0
    for val in values(data.sdd_lookup)
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_lb"]
                w_en_min_pr_ind += 1
                for t in periods
                    if w[1] + ε_time < get_as(dt, t)[2] && get_as(dt, t)[2] <= w[2] + ε_time
                        push!(T_w_en_min_pr, (w_en_min_pr_ind=w_en_min_pr_ind, j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                        j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                        j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                        t = t,
                        dt = dt[t]))
                    end
                end
            end
        end
    end

    T_w_en_min_cs = Vector{@NamedTuple{w_en_min_cs_ind::Int, j::Int, j_prcs::Int, j_cs::Int, t::Int, dt::Float64}}()
    w_en_min_cs_ind = 0
    for val in values(data.sdd_lookup)
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first)
            for w in val["energy_req_lb"]
                w_en_min_cs_ind += 1
                for t in periods
                    if w[1] + ε_time < get_as(dt, t)[2] && get_as(dt, t)[2] <= w[2] + ε_time
                        push!(T_w_en_min_cs, (w_en_min_cs_ind=w_en_min_cs_ind, j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1, 
                        j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                        j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                        t = t,
                        dt = dt[t]))
                    end
                end
            end
        end
    end

    p_jtm_flattened_pr = Vector{@NamedTuple{flat_k::Int, j::Int, j_prcs::Int, j_pr::Int, t::Int, m::Int, c_en::Float64, p_max::Float64}}()
    flat_k=1
    for (pc_idx, pc) in pairs(cost_vector_pr)
        j=pc.j
        j_prcs=pc.j_prcs
        j_pr = pc.j_pr
        for (t, cost_t) in enumerate(pc.cost)
            for (m, cost_tm) in enumerate(cost_t)
                c_en, p_max = cost_tm
                push!(p_jtm_flattened_pr, (flat_k=flat_k, j=j, j_prcs=j_prcs, j_pr=pc.j_pr, t=t, m=m, c_en=c_en, p_max=p_max))
                flat_k+=1
            end
        end
    end

    p_jtm_flattened_cs = Vector{@NamedTuple{flat_k::Int, j::Int, j_prcs::Int, j_cs::Int, t::Int, m::Int, c_en::Float64, p_max::Float64}}()
    flat_k=1
    for (pc_idx, pc) in pairs(cost_vector_cs)
        j=pc.j
        j_prcs=pc.j_prcs
        j_cs = pc.j_cs
        for (t, cost_t) in enumerate(pc.cost)
            for (m, cost_tm) in enumerate(cost_t)
                c_en, p_max = cost_tm
                push!(p_jtm_flattened_cs, (flat_k=flat_k, j=j, j_prcs=j_prcs, j_cs=pc.j_cs, t=t, m=m, c_en=c_en, p_max=p_max))
                flat_k+=1
            end
        end
    end

    jtk_ln_flattened = Vector{@NamedTuple{flat_jtk_ln::Int, ctg::Int, j::Int, j_ac::Int, j_ln::Int, to_bus::Int, fr_bus::Int, b_sr::Float64, s_max_ctg::Float64, u_on::Int, t::Int, dt::Float64}}()
    flat_jtk_ln = 1
    for ctg in data_json["reliability"]["contingency"], t in periods
        for component in ctg["components"]
            for val in values(data.ac_line_lookup) 
                if val["uid"] != component
                    for uc in uc_data["time_series_output"]["ac_line"]
                        if val["uid"] == uc["uid"]
                            r = val["r"]
                            x = val["x"]
                            push!(jtk_ln_flattened, (flat_jtk_ln=flat_jtk_ln, ctg = parse(Int, match(r"\d+", ctg["uid"]).match)+1, j = parse(Int, match(r"\d+", val["uid"]).match)+1, j_ac = parse(Int, match(r"\d+", val["uid"]).match)+1, 
                            j_ln = parse(Int, match(r"\d+", val["uid"]).match)+1, to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1,
                            fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1, b_sr = -x / (x^2 + r^2), s_max_ctg = val["mva_ub_em"], u_on=uc["on_status"][t], t=t, dt = dt[t]))
                            flat_jtk_ln += 1
                        end
                    end
                end
            end
        end
    end

    

    jtk_xf_flattened = Vector{@NamedTuple{flat_jtk_xf::Int, ctg::Int, j::Int, j_ac::Int, j_xf::Int, to_bus::Int, fr_bus::Int, b_sr::Float64, s_max_ctg::Float64, u_on::Int, t::Int, dt::Float64}}()
    flat_jtk_xf = 1
    for ctg in data_json["reliability"]["contingency"], t in periods
        for component in ctg["components"]
            for val in values(data.twt_lookup) 
                if val["uid"] != component
                    for uc in uc_data["time_series_output"]["two_winding_transformer"]
                        if val["uid"] == uc["uid"]
                            r = val["r"]
                            x = val["x"]
                            push!(jtk_xf_flattened, (flat_jtk_xf=flat_jtk_xf, ctg = parse(Int, match(r"\d+", ctg["uid"]).match)+1, j = parse(Int, match(r"\d+", val["uid"]).match)+1+ L_J_ln, j_ac = parse(Int, match(r"\d+", val["uid"]).match)+1+ L_J_ln, 
                            j_xf = parse(Int, match(r"\d+", val["uid"]).match)+1, to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1,
                            fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1, b_sr = -x / (x^2 + r^2), s_max_ctg = val["mva_ub_em"], u_on=uc["on_status"][t], t=t, dt = dt[t]))
                            flat_jtk_xf += 1
                        end
                    end
                end
            end
        end
    end

    jtk_dc_flattened = Vector{@NamedTuple{flat_jtk_dc::Int, ctg::Int, j::Int, j_dc::Int, to_bus::Int, fr_bus::Int, t::Int, dt::Float64}}()
    flat_jtk_dc = 1
    for ctg in data_json["reliability"]["contingency"], t in periods
        for component in ctg["components"]
            for val in values(data.dc_line_lookup) 
                if val["uid"] != component
                    push!(jtk_dc_flattened, (flat_jtk_dc=flat_jtk_dc, ctg = parse(Int, match(r"\d+", ctg["uid"]).match)+1, j = parse(Int, match(r"\d+", val["uid"]).match)+1+ L_J_ac, 
                    j_dc = parse(Int, match(r"\d+", val["uid"]).match)+1, to_bus = parse(Int, match(r"\d+", val["to_bus"]).match)+1,
                    fr_bus = parse(Int, match(r"\d+", val["fr_bus"]).match)+1, t=t, dt = dt[t]))
                    flat_jtk_dc += 1
                end
            end
        end
    end

    empty_vpd = Vector{NamedTuple{(:j, :j_ac, :j_xf, :phi_min, :phi_max, :t), Tuple{Int64, Int64, Int64, Float64, Float64, Int64}}}()
    empty_fpd = Vector{NamedTuple{(:j, :j_ac, :j_xf, :phi_o, :t), Tuple{Int64, Int64, Int64, Float64, Int64}}}()
    empty_vwr = Vector{NamedTuple{(:j, :j_ac, :j_xf, :tau_min, :tau_max, :t), Tuple{Int64, Int64, Int64, Float64, Float64, Int64}}}()
    empty_fwr = Vector{NamedTuple{(:j, :j_ac, :j_xf, :tau_o, :t), Tuple{Int64, Int64, Int64, Float64, Int64}}}()

    sc_time_data = (
        ;
        periods = periods,

        c_p = [data.violation_cost["p_bus_vio_cost"]],
        c_q = [data.violation_cost["q_bus_vio_cost"]],
        c_s = [data.violation_cost["s_vio_cost"]],
        c_e = [data.violation_cost["e_vio_cost"]],
        busarray = [(;b..., t=t, dt=dt[t]) for b in sc_data.bus, t in periods],
        k_busarray = [(;b..., t=t, k=k) for b in sc_data.bus, t in periods, k in 1:length(data_json["reliability"]["contingency"])],
        shuntarray = [
            (;s..., t=t, u_sh = uc["step"][t])
            for s in sc_data.shunt, t in periods
            for uc in uc_data["time_series_output"]["shunt"]
            if s.uid == uc["uid"]
                ],
        k_shuntarray = [(;b..., t=t, k=k) for b in sc_data.shunt, t in periods, k in 1:length(data_json["reliability"]["contingency"])],
       

        preservearray = [(;n=r.n, n_p=r.n_p, uid=r.uid, c_rgu=r.c_rgu, c_rgd=r.c_rgd, c_scr=r.c_scr, c_nsc=r.c_nsc, c_rru=r.c_rru, c_rrd=r.c_rrd,
        σ_rgu=r.σ_rgu, σ_rgd=r.σ_rgd, σ_scr=r.σ_scr, σ_nsc=r.σ_nsc, p_rru_min=r.p_rru_min[t], p_rrd_min=r.p_rrd_min[t],
        t=t, dt=dt[t]) for r in sc_data.active_reserve, t in periods],

        qreservearray = [(;n=q.n, n_q=q.n_q, uid=q.uid, c_qru=q.c_qru, c_qrd=q.c_qrd, q_qru_min=q.q_qru_min[t], q_qrd_min=q.q_qrd_min[t], t=t, dt = dt[t])
        for q in sc_data.reactive_reserve, t in periods],

        k_prarray = [(;j_prcs=b.j_prcs, j_pr=b.j_pr, bus = b.bus, uid=b.uid, t=t, k=k) for b in sc_data.prod, t in periods, k in 1:length(data_json["reliability"]["contingency"])],
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

        prarray_pqbounds = isempty(val for val in values(data.sdd_lookup) if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1) ? 
                            empty_data = Vector{NamedTuple{(:j, :jprcs, :j_pr, :u_on, :sum2_T_supc_pr_jt, :sum2_T_sdpc_pr_jt, :beta_max, :beta_min, :q_max_p0, :q_min_p0, :t), Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Int64}}}() : [
                            (;j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                            u_on = uc["on_status"][t],
                            sum2_T_supc_pr_jt=sum2_T_supc_pr[p.j_pr, t], 
                            sum2_T_sdpc_pr_jt=sum2_T_sdpc_pr[p.j_pr, t], 
                            beta_max = val["beta_ub"],
                            beta_min = val["beta_lb"],
                            q_max_p0 = val["q_0_ub"],
                            q_min_p0 = val["q_0_lb"],
                            t=t
        )
        for val in values(data.sdd_lookup), t in periods
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        prarray_pqe = isempty(val for val in values(data.sdd_lookup) if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1) ? 
                            empty_data = Vector{NamedTuple{(:j, :jprcs, :j_pr, :u_on, :sum2_T_supc_pr_jt, :sum2_T_sdpc_pr_jt, :beta, :q_p0, :t), Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Float64, Float64, Int64}}}() : [
                            (;j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_pr=get_j_pr(val["uid"], L_J_pr, L_J_cs, producers_first),
                            u_on = uc["on_status"][t],
                            sum2_T_supc_pr_jt=sum2_T_supc_pr[p.j_pr, t], 
                            sum2_T_sdpc_pr_jt=sum2_T_sdpc_pr[p.j_pr, t], 
                            beta = val["beta"],
                            q_p0 = val["q_0"],
                            t=t
        )
        for val in values(data.sdd_lookup), t in periods
        if is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_linear_cap"]==1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        k_csarray = [(;j_prcs=b.j_prcs, j_cs=b.j_cs, bus = b.bus, uid=b.uid, t=t, k=k) for b in sc_data.cons, t in periods, k in 1:length(data_json["reliability"]["contingency"])],

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

        csarray_pqbounds = isempty(val for val in values(data.sdd_lookup) if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1) ? 
                            empty_data = Vector{NamedTuple{(:j, :jprcs, :j_cs, :u_on, :sum2_T_supc_cs_jt, :sum2_T_sdpc_cs_jt, :beta_max, :beta_min, :q_max_p0, :q_min_p0, :t), Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64, Int64}}}() : [
                            (;j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                            j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            u_on = uc["on_status"][t],
                            sum2_T_supc_cs_jt=sum2_T_supc_cs[p.j_cs, t], 
                            sum2_T_sdpc_cs_jt=sum2_T_sdpc_cs[p.j_cs, t], 
                            beta_max = val["beta_ub"],
                            beta_min = val["beta_lb"],
                            q_max_p0 = val["q_0_ub"],
                            q_min_p0 = val["q_0_lb"],
                            t=t
        )
        for val in values(data.sdd_lookup), t in periods
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1
        for uc in uc_data["time_series_output"]["simple_dispatchable_device"]
        if p.uid == uc["uid"]],

        csarray_pqe = isempty(val for val in values(data.sdd_lookup) if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_bound_cap"]==1) ? 
                            empty_data = Vector{NamedTuple{(:j, :jprcs, :j_cs, :u_on, :sum2_T_supc_cs_jt, :sum2_T_sdpc_cs_jt, :beta, :q_p0, :t), Tuple{Int64, Int64, Int64, Int64, Int64, Int64, Float64, Float64, Int64}}}() : [
                            (;j = parse(Int, match(r"\d+", val["uid"]).match) + L_J_br + 1,
                            j_cs=get_j_cs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            j_prcs=get_j_prcs(val["uid"], L_J_pr, L_J_cs, producers_first),
                            u_on = uc["on_status"][t],
                            sum2_T_supc_cs_jt=sum2_T_supc_cs[p.j_cs, t], 
                            sum2_T_sdpc_cs_jt=sum2_T_sdpc_cs[p.j_cs, t], 
                            beta = val["beta"],
                            q_p0 = val["q_0"],
                            t=t
        )
        for val in values(data.sdd_lookup), t in periods
        if !is_pr(val["uid"], L_J_pr, L_J_cs, producers_first) && val["q_linear_cap"]==1
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

        fpdarray = isempty(sc_data.fpd) ? empty_data = empty_fpd : [(;b..., t=t) for b in sc_data.fpd, t in periods],
        fwrarray = isempty(sc_data.fwr) ? empty_data = empty_fwr : [(;b..., t=t) for b in sc_data.fwr, t in periods],
        vpdarray = isempty(sc_data.vpd) ? empty_data = empty_vpd : [(;b..., t=t) for b in sc_data.vpd, t in periods],
        vwrarray = isempty(sc_data.vwr) ? empty_data = empty_vwr : [(;b..., t=t) for b in sc_data.vwr, t in periods],
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

        tk_index = [(t=t, k=k) for t in periods, k in 1:length(data_json["reliability"]["contingency"])],

        p_jtm_flattened_pr=p_jtm_flattened_pr,
        p_jtm_flattened_cs=p_jtm_flattened_cs,
        p_jtm_pr_uvar = [b.p_max for b in p_jtm_flattened_pr],
        p_jtm_cs_uvar = [b.p_max for b in p_jtm_flattened_cs],
        jtk_ln_flattened=jtk_ln_flattened,
        jtk_xf_flattened=jtk_xf_flattened,
        jtk_dc_flattened=jtk_dc_flattened,

    )

    lengths = (L_J_xf, L_J_ln, L_J_ac, L_J_dc, L_J_br, L_J_cs,
    L_J_pr, L_J_cspr, L_J_sh, I, L_T, L_N_p, L_N_q, L_W_en_min_pr, L_W_en_min_cs, L_W_en_max_pr, L_W_en_max_cs, K)

    return sc_time_data, lengths, producers_first
end

function save_go3_solution(uc_filename, solution_name, result, vars, lengths, producers_first)
    uc_data = JSON.parsefile(uc_filename)
    (L_J_xf, L_J_ln, L_J_ac, L_J_dc, L_J_br, L_J_cs,
    L_J_pr, L_J_cspr, L_J_sh, I, L_T, L_N_p, L_N_q, L_W_en_min_pr, L_W_en_min_cs, L_W_en_max_pr, L_W_en_max_cs, K) = lengths
    #Update simple dispatchable devices
    for line in uc_data["time_series_output"]["simple_dispatchable_device"]
        raw_uid = parse(Int, match(r"\d+", line["uid"]).match)
        solution_index = raw_uid + 1 #This corresponds to j_prcs
        if !is_pr(raw_uid, L_J_pr, L_J_cs, producers_first)
            #This section corresponds to consuming devices
            if producers_first
                solution_index -= L_J_pr
            end
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
            if !producers_first
                solution_index -= L_J_cs
            end
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
        solution_index = parse(Int, match(r"\d+", line["uid"]).match) + 1 #This corresponds to j_xf
        line["tm"] = Array(solution(result, vars.τ_jt_xf))[solution_index,:]
        line["ta"] = Array(solution(result, vars.φ_jt_xf))[solution_index,:] 
    end
    #ac line flows can be inferred from voltage and angle levels

    #Update DC lines
    for line in uc_data["time_series_output"]["dc_line"]
        solution_index = parse(Int, match(r"\d+", line["uid"]).match) + 1 #This corresponds to j_dc
        line["qdc_fr"] = Array(solution(result, vars.q_jt_fr_dc))[solution_index,:]
        line["pdc_fr"] = Array(solution(result, vars.p_jt_fr_dc))[solution_index,:] #pdc_to is implied from energy conservation
        line["qdc_to"] = Array(solution(result, vars.q_jt_to_dc))[solution_index,:]
    end

    #Update buses
    for line in uc_data["time_series_output"]["bus"]
        solution_index = parse(Int, match(r"\d+", line["uid"]).match) + 1 #This corresponds to i
        line["vm"] = Array(solution(result, vars.v_it))[solution_index,:]
        line["va"] = Array(solution(result, vars.θ_it))[solution_index,:]
    end

    open(solution_name, "w") do io
        JSON.print(io, uc_data, 4)
    end
end