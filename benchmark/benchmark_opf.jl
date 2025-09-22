
using MadNLPHSL, NLPModelsIpopt, NLPModels, LinearAlgebra, CSV, DataFrames, PrettyTables, Printf, Plots, SolverBenchmark, MadNCL, HybridKKT, DataStructures, CUDA, ExaModelsPower, ExaModels, MadNLP, MadNLPGPU, PowerModels, JuMP
using NLPModelsJuMP
using PrettyTables: tf_latex_booktabs, LatexTableFormat

sample_cases = ["pglib_opf_case3_lmbd", 
"pglib_opf_case5_pjm",
"pglib_opf_case14_ieee",
"pglib_opf_case24_ieee_rts",
"pglib_opf_case39_epri",
"pglib_opf_case89_pegase",
"pglib_opf_case197_snem",
"pglib_opf_case500_goc",
"pglib_opf_case1888_rte",
"pglib_opf_case2736sp_k",
"pglib_opf_case2848_rte",
"pglib_opf_case3375wp_k",
"pglib_opf_case4917_goc",
"pglib_opf_case9241_pegase",
"pglib_opf_case19402_goc",
"pglib_opf_case78484_epigrids",]
small_sample_cases = [
"pglib_opf_case3_lmbd", 
"pglib_opf_case5_pjm",
"pglib_opf_case14_ieee",
"pglib_opf_case24_ieee_rts",
"pglib_opf_case30_as",
"pglib_opf_case30_ieee",]
cases = [
"pglib_opf_case3_lmbd", 
"pglib_opf_case5_pjm",
"pglib_opf_case14_ieee",
"pglib_opf_case24_ieee_rts",
"pglib_opf_case30_as",
"pglib_opf_case30_ieee",
"pglib_opf_case39_epri",
"pglib_opf_case57_ieee",
"pglib_opf_case60_c",
"pglib_opf_case73_ieee_rts",
"pglib_opf_case89_pegase",
"pglib_opf_case118_ieee",
"pglib_opf_case162_ieee_dtc",
"pglib_opf_case179_goc",
"pglib_opf_case197_snem",
"pglib_opf_case200_activ",
"pglib_opf_case240_pserc",
"pglib_opf_case300_ieee",
"pglib_opf_case500_goc",
"pglib_opf_case588_sdet",
"pglib_opf_case793_goc",
"pglib_opf_case1354_pegase",
"pglib_opf_case1803_snem",
"pglib_opf_case1888_rte",
"pglib_opf_case1951_rte",
"pglib_opf_case2000_goc",
"pglib_opf_case2312_goc",
"pglib_opf_case2383wp_k",
"pglib_opf_case2736sp_k",
"pglib_opf_case2737sop_k",
"pglib_opf_case2742_goc",
"pglib_opf_case2746wop_k",
"pglib_opf_case2746wp_k",
"pglib_opf_case2848_rte",
"pglib_opf_case2853_sdet",
"pglib_opf_case2868_rte",
"pglib_opf_case2869_pegase",
"pglib_opf_case3012wp_k",
"pglib_opf_case3022_goc",
"pglib_opf_case3120sp_k",
"pglib_opf_case3375wp_k",
"pglib_opf_case3970_goc",
"pglib_opf_case4020_goc",
"pglib_opf_case4601_goc",
"pglib_opf_case4619_goc",
"pglib_opf_case4661_sdet",
"pglib_opf_case4837_goc",
"pglib_opf_case4917_goc",
"pglib_opf_case5658_epigrids",
"pglib_opf_case6468_rte",
"pglib_opf_case6470_rte",
"pglib_opf_case6495_rte",
"pglib_opf_case6515_rte",
"pglib_opf_case7336_epigrids",
"pglib_opf_case8387_pegase",
"pglib_opf_case9241_pegase",
"pglib_opf_case9591_goc",
"pglib_opf_case10000_goc",
"pglib_opf_case10192_epigrids",
"pglib_opf_case10480_goc",
"pglib_opf_case13659_pegase",
"pglib_opf_case19402_goc",
"pglib_opf_case20758_epigrids",
"pglib_opf_case24464_goc",
"pglib_opf_case30000_goc",
"pglib_opf_case78484_epigrids",]

function termination_code(status::MadNLP.Status)
    if status == MadNLP.SOLVE_SUCCEEDED
        return " "
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return "a"
    elseif status == MadNLP.DIVERGING_ITERATES || status == MadNLP.DIVERGING_ITERATES
        return "i"
    elseif status == MadNLP.RESTORATION_FAILED
        return "r"
    else
        return "f"
    end
end

function termination_code(status::Symbol)
    if status == :Solve_Succeeded
        return " "
    elseif status == :Solved_To_Acceptable_Level
        return "a"
    elseif status == :Infeasible_Problem_Detected || status == :Diverging_Iterates
        return "i"
    elseif status == :Restoration_Failed
        return "r"
    else
        return "f"
    end
end

function evaluate(m, result)
    constraints = similar(result.solution, m.meta.ncon)
    NLPModels.cons!(m, result.solution, constraints)
    return max(
        norm(min.(result.solution .- m.meta.lvar, 0), Inf),
        norm(min.(m.meta.uvar .- result.solution, 0), Inf),
        norm(min.(constraints .- m.meta.lcon, 0), Inf),
        norm(min.(m.meta.ucon .- constraints, 0), Inf)
    )
end

function ipopt_stats(fname)
    output = read(fname, String)
    iter = parse(Int, split(split(output, "Number of Iterations....:")[2], "\n")[1])
    i = parse(Float64,split(split(output, "Total seconds in IPOPT (w/o function evaluations)    =")[2], "\n")[1])
    ad = parse(Float64,split(split(output, "Total seconds in NLP function evaluations            =")[2], "\n")[1])
    tot = i + ad
    return iter, tot, ad
end


# after you assemble `methods` and `subs`
function group_boundaries(methods, subs)
    idx  = Int[0, 1]             # columns AFTER which a line is inserted
    col  = 1                 # 1st column = "Case"

    for m in methods
        col += length(subs[m])
        push!(idx, col)      # boundary just after each group
    end

    return idx               # e.g. [4, 7, 9] for groups of size 3,3,2
end



function generate_tex_opf(opf_results::Dict; filename="benchmark_results_opf.tex")

    df_top = opf_results[:top]
    df_lifted_kkt = opf_results[:lifted_kkt]
    df_hybrid_kkt = opf_results[:hybrid_kkt]
    df_madncl = opf_results[:madncl]
    df_ma27 = opf_results[:ma27]
    df_ma86 = opf_results[:ma86]
    #df_ma97 = opf_results[:ma97]
    

    methods = ["MadNLP+LiftedKKT (GPU)", "MadNLP+HybridKKT (GPU)", "MadNCL (GPU)",
            "Ipopt+Ma27 (CPU)","Ipopt+Ma86 (CPU)",]#"Ipopt+Ma97 (CPU)"]
    subs = Dict(
        "MadNLP+LiftedKKT (GPU)" => [:iter, :soltime, :inittime, :adtime,
                          :lintime, :termination, :obj, :cvio],
        "MadNLP+HybridKKT (GPU)" => [:iter, :soltime, :inittime, :adtime,
                            :lintime, :termination, :obj, :cvio],
        "MadNCL (GPU)" => [:iter, :soltime, :inittime, :adtime,
                          :lintime, :termination, :obj, :cvio],
        "Ipopt+Ma27 (CPU)" => [:iter, :soltime, :adtime,
                          :termination, :obj, :cvio],
        "Ipopt+Ma86 (CPU)" => [:iter, :soltime, :adtime,
                          :termination, :obj, :cvio],
        "MadNLP+Ma86 (CPU)" => [:iter, :soltime, :inittime, :adtime,
                          :lintime, :termination, :obj, :cvio],
        #"Ipopt+Ma97 (CPU)" => [:iter, :soltime, :adtime,
        #                  :termination, :obj, :cvio],
    )

    format_val(field, val) =
        (val === missing || val === nothing) ? missing :
        !(val isa Number) ? string(val) :
        field == :iter ? string(Int(round(val))) :
        field in [:obj, :cvio] ? @sprintf("%.6e", val) :
        @sprintf("%.3e", round(val, sigdigits=4))

    format_k(val) = isnothing(val) || val === missing ? missing : @sprintf("%.1fk", val / 1000)

    rows = Any[]
    raw_rows = Any[]
    for (i, row_top) in enumerate(eachrow(df_top))
        case = row_top.case_name
        clean_case = replace(case,
            r"^(api/|sad/)" => "",
            r"pglib_opf_case" => "",
            r"\.m$" => ""
        )
        row = Any[clean_case, format_k(row_top.nvar), format_k(row_top.ncon)]
        raw_row = Any[clean_case, row_top.nvar, row_top.ncon]

        methods = ["MadNLP+LiftedKKT (GPU)", "MadNLP+HybridKKT (GPU)", "MadNCL (GPU)",
            "Ipopt+Ma27 (CPU)","MadNLP+Ma86 (CPU)",]#"Ipopt+Ma97 (CPU)"]
        for (df, method) in [(df_lifted_kkt, "MadNLP+LiftedKKT (GPU)"), (df_hybrid_kkt, "MadNLP+HybridKKT (GPU)"),
                            (df_madncl, "MadNCL (GPU)"), (df_ma27, "Ipopt+Ma27 (CPU)"),
                            (df_ma86, "MadNLP+Ma86 (CPU)"),]# (df_ma97, "Ipopt+Ma97 (CPU)")]
            df_row = df[i, :]
            for field in subs[method]
                val = get(df_row, field, missing)
                push!(row, format_val(field, val))
                push!(raw_row, val)
            end
        end

        push!(rows, row)
        push!(raw_rows, raw_row)
    end


    table_data = permutedims(reduce(hcat, rows))
    nrows = length(rows)
    hlines = vcat(0, 1, collect(6:5:nrows), nrows+1)

    h_top    = ["Case", "nvars", "ncons"]
    h_bottom = ["",     "",      ""]

    for m in methods
        n = length(subs[m])
        push!(h_top, string(m))
        append!(h_top, fill("", n-1))
        append!(h_bottom, string.(subs[m]))
    end

    function group_boundaries(methods, subs)
        idx = Int[0, 1, 2, 3]
        col = 3
        for m in methods
            col += length(subs[m])
            push!(idx, col)
        end
        return idx
    end

    vlines = group_boundaries(methods, subs)

    tex_filename = replace(filename, r"\.csv$" => ".tex")
    open(tex_filename, "w") do io
        pretty_table(
            io, table_data;
            header = (h_top, h_bottom),
            backend = Val(:latex),
            tf = tf_latex_default,
            alignment = :c,
            vlines = vlines,
            hlines = hlines
        )
    end

    # Text file
    txt_filename = replace(filename, r"\.csv$" => ".txt")
    open(txt_filename, "w") do io
        pretty_table(
            io, table_data;
            header = (h_top, h_bottom),
            backend = Val(:text),
            alignment = :c
        )
    end

    # Raw CSV
    csv_filename = filename
    flat_header = vcat(["Case", "nvars", "ncons"], vcat([
        string(m, "_", f) for m in methods for f in subs[m]
    ]))
    df = DataFrame([Symbol(h) => col for (h, col) in zip(flat_header, eachcol(permutedims(reduce(hcat, raw_rows))))])
    CSV.write(csv_filename, df)


    # Full comparison
    selected = Dict(k => opf_results[k] for k in [:lifted_kkt, :hybrid_kkt, :madncl, :ma27, :ma86,])# :ma97])
    p = performance_profile(selected, df -> df.soltime)
    Plots.svg(p, replace(filename, r"\.csv$" => ""))

    small_list = Int64[]
    med_list= Int64[]
    large_list = Int64[]

    for (i, row) in enumerate(eachrow(opf_results[:top]))
        if row.nvar <= 2000
            push!(small_list, i)
        elseif row.nvar <= 20000
            push!(med_list, i)
        else
            push!(large_list, i)
        end
    end
    
    # Small: nvar < 2000
    ordered_keys = [:lifted_kkt, :hybrid_kkt, :madncl, :ma27, :ma86,]# :ma97]

    selected = OrderedDict(
        k => filter(row -> row.id in small_list, opf_results[k])
        for k in ordered_keys
        if haskey(opf_results, k)
    )
    # Now build an ordered list of Pairs
    if !isempty(selected[:lifted_kkt])
        p = performance_profile(selected, df -> df.soltime)
        Plots.svg(p, replace(filename, r"\.csv$" => "_small"))
    end


    # Medium: 2000 ≤ nvar ≤ 20000
    selected = OrderedDict(
        k => filter(row -> row.id in med_list, opf_results[k])
        for k in ordered_keys
        if haskey(opf_results, k)
    )
    # Now build an ordered list of Pairs
    if !isempty(selected[:lifted_kkt])
        p = performance_profile(selected, df -> df.soltime)
        Plots.svg(p, replace(filename, r"\.csv$" => "_medium"))
    end

    # Large: nvar > 20000
    selected = OrderedDict(
        k => filter(row -> row.id in large_list, opf_results[k])
        for k in ordered_keys
        if haskey(opf_results, k)
    )
    # Now build an ordered list of Pairs
    if !isempty(selected[:lifted_kkt])
        p = performance_profile(selected, df -> df.soltime)
        Plots.svg(p, replace(filename, r"\.csv$" => "_large"))
    end


    # Log-log scatterplot of speedup vs nvar

    baseline = df_ma27
    n = nrow(df_top)

    scatter_data = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

    for (method, df) in [
        ("MadNLP+LiftedKKT (GPU)", df_lifted_kkt),
        ("MadNLP+HybridKKT (GPU)", df_hybrid_kkt),
        ("MadNCL (GPU)", df_madncl),
        ("Ipopt+Ma27 (CPU)", df_ma27),
        ("MadNLP+Ma86 (CPU)", df_ma86),
        #("Ipopt+Ma97 (CPU)", df_ma97)
    ]
        nv = Float64[]
        speedup = Float64[]
        for i in 1:n
            t_base = get(baseline[i, :], :soltime, missing)
            t = get(df[i, :], :soltime, missing)
            nv_i = get(df_top[i, :], :nvar, missing)

            if t !== missing && t_base !== missing && nv_i !== missing && t > 0 && t_base > 0
                push!(nv, float(nv_i))
                push!(speedup, t_base / t)
            end
        end
        scatter_data[method] = (nv, speedup)
    end

    p = plot(
        xlabel = "nvar", ylabel = "Speedup vs. Ma27",
        xscale = :log10, yscale = :log10,
        legend = :topleft, title = "Speedup vs. Problem Size",
        markerstrokewidth = 0
    )

    for (method, (nv, speedup)) in scatter_data
        scatter!(p, nv, speedup; label = method, ms=4)
    end

    svg_speedup = replace(filename, r"\.csv$" => "_speedup_vs_ma27.svg")
    savefig(p, svg_speedup)



end



function save_opf_results(opf_results, path)
    dfs = Dict{Symbol, DataFrame}()

    for (key, df) in opf_results
        df_ = copy(df)
        solver_name = string(key)

        # Create renamed column mapping with suffix
        new_names = [Symbol(string(name), "_", solver_name) for name in names(df_)]
        rename!(df_, names(df_) .=> new_names)

        dfs[key] = df_
    end

    all_dfs = hcat(values(dfs)...)
    CSV.write(path, all_dfs)
end

function merge_static_data(gpu_filename, cpu_filename, save_folder)
    gpu_results = CSV.read(gpu_filename, DataFrame)
    cpu_results = CSV.read(cpu_filename, DataFrame)

    df_top = select(gpu_results, r"_top$")
    rename!(df_top, Symbol.(replace.(String.(names(df_top)), "_top" => "")))
    df_lifted_kkt = select(gpu_results, r"lifted_kkt$")
    rename!(df_lifted_kkt, Symbol.(replace.(String.(names(df_lifted_kkt)), "_lifted_kkt" => "")))
    df_hybrid_kkt = select(gpu_results, r"hybrid_kkt$")
    rename!(df_hybrid_kkt, Symbol.(replace.(String.(names(df_hybrid_kkt)), "_hybrid_kkt" => "")))
    df_madncl = select(gpu_results, r"madncl$")
    rename!(df_madncl, Symbol.(replace.(String.(names(df_madncl)), "_madncl" => "")))
    df_ma27 = select(cpu_results, r"ma27$")
    rename!(df_ma27, Symbol.(replace.(String.(names(df_ma27)), "_ma27" => "")))
    df_ma86 = select(cpu_results, r"ma86$")
    rename!(df_ma86, Symbol.(replace.(String.(names(df_ma86)), "_ma86" => "")))

    opf_results = Dict(:top => df_top,
                :lifted_kkt => df_lifted_kkt,
                :hybrid_kkt => df_hybrid_kkt,
                :madncl => df_madncl,
                :ma27 => df_ma27,
                :ma86 => df_ma86)
    
    generate_tex_opf(opf_results; filename = save_folder*replace(replace(gpu_filename, "_GPU" => ""), "saved_raw_data/" => ""))
end
    
function summary_table(filenames, max_wall_time, delta, save_prefix; mp=false)
    
    df = DataFrame(
        tol = Any[],
        solver = Any[],
        small_count = Int[],
        small_time = Float64[],
        small_cvio = Float64[],
        med_count = Int[],
        med_time = Float64[],
        med_cvio = Float64[],
        large_count = Int[],
        large_time = Float64[],
        large_cvio = Float64[],
        total_count = Int[],
        total_time = Float64[],
        total_cvio = Float64[]
    )


    n_small = 0
    n_med = 0
    n_large = 0
    n_total = 0 
    for file in filenames
        results = CSV.read(file, DataFrame)

        if mp
            # find all *_termination columns
            term_cols = filter(c -> endswith(c, "_termination"), names(results))

            # keep rows that have at least one " " or "a" in termination columns
            results = filter(row -> any(row[c] in (" ", "a") for c in term_cols), results)
        end

        df_lifted_kkt = Dict(
            :small_count => 0, :small_time => Float64(1), :small_cvio => Float64(1),
            :med_count => 0, :med_time => Float64(1), :med_cvio => Float64(1),
            :large_count => 0, :large_time => Float64(1), :large_cvio => Float64(1),
            :total_count => 0, :total_time => Float64(1), :total_cvio => Float64(1)
        )
        df_hybrid_kkt = Dict(
            :small_count => 0, :small_time => Float64(1), :small_cvio => Float64(1),
            :med_count => 0, :med_time => Float64(1), :med_cvio => Float64(1),
            :large_count => 0, :large_time => Float64(1), :large_cvio => Float64(1),
            :total_count => 0, :total_time => Float64(1), :total_cvio => Float64(1)
        )
        df_madncl = Dict(
            :small_count => 0, :small_time => Float64(1), :small_cvio => Float64(1),
            :med_count => 0, :med_time => Float64(1), :med_cvio => Float64(1),
            :large_count => 0, :large_time => Float64(1), :large_cvio => Float64(1),
            :total_count => 0, :total_time => Float64(1), :total_cvio => Float64(1)
        )
        df_ma27 = Dict(
            :small_count => 0, :small_time => Float64(1), :small_cvio => Float64(1),
            :med_count => 0, :med_time => Float64(1), :med_cvio => Float64(1),
            :large_count => 0, :large_time => Float64(1), :large_cvio => Float64(1),
            :total_count => 0, :total_time => Float64(1), :total_cvio => Float64(1)
        )
        df_ma86 = Dict(
            :small_count => 0, :small_time => Float64(1), :small_cvio => Float64(1),
            :med_count => 0, :med_time => Float64(1), :med_cvio => Float64(1),
            :large_count => 0, :large_time => Float64(1), :large_cvio => Float64(1),
            :total_count => 0, :total_time => Float64(1), :total_cvio => Float64(1)
        )
        
        matched_solvers = Dict("MadNLP+LiftedKKT (GPU)" => df_lifted_kkt,
                    "MadNLP+HybridKKT (GPU)" => df_hybrid_kkt,
                    "MadNCL (GPU)" => df_madncl,
                    "Ipopt+Ma27 (CPU)" => df_ma27,
                    "MadNLP+Ma86 (CPU)" => df_ma86)

        n_small = 0
        n_med = 0
        n_large = 0
        n_total = nrow(results)
        for row in eachrow(results)
            if row["nvars"] < 2000
                count_label = :small_count
                time_label = :small_time
                cvio_label = :small_cvio
                n_small += 1
            elseif row["nvars"] < 20000
                count_label = :med_count
                time_label = :med_time
                cvio_label = :med_cvio
                n_med += 1
            else
                count_label = :large_count
                time_label = :large_time
                cvio_label = :large_cvio
                n_large += 1
            end

            for solver_name in keys(matched_solvers)
                termination_label = solver_name * "_termination"
                if row[termination_label] == " " || row[termination_label] == "a"
                    matched_solvers[solver_name][count_label] += 1
                    matched_solvers[solver_name][:total_count] += 1
                    matched_solvers[solver_name][time_label] = matched_solvers[solver_name][time_label]*(row[solver_name*"_soltime"] + delta)
                    matched_solvers[solver_name][:total_time] = matched_solvers[solver_name][:total_time]*(row[solver_name*"_soltime"] + delta)
                    if typeof(row[solver_name*"_cvio"]) != Float64
                        matched_solvers[solver_name][cvio_label] = matched_solvers[solver_name][cvio_label]*(parse(Float64, row[solver_name*"_cvio"]) + delta)
                        matched_solvers[solver_name][:total_cvio] = matched_solvers[solver_name][:total_cvio]*(parse(Float64, row[solver_name*"_cvio"]) + delta)
                    else
                        matched_solvers[solver_name][cvio_label] = matched_solvers[solver_name][cvio_label]*(row[solver_name*"_cvio"] + delta)
                        matched_solvers[solver_name][:total_cvio] = matched_solvers[solver_name][:total_cvio]*(row[solver_name*"_cvio"] + delta)
                    end
                else
                    matched_solvers[solver_name][time_label] = matched_solvers[solver_name][time_label]*(max_wall_time + delta)
                    matched_solvers[solver_name][:total_time] = matched_solvers[solver_name][:total_time]*(max_wall_time + delta)
                end
            end  
        end
        n = nrow(results)
        n_dict = Dict(:small_time => n_small, :med_time => n_med, :large_time => n_large, :total_time => n)
        

        m = match(r"tol_(\d+e\d+)", file)

        if m !== nothing
            val = m.captures[1]              # "1e4"
            val = replace(val, "e" => "e-")  # insert the missing minus
            tol = parse(Float64, val)  # numeric value: 1.0e-4
        end

        for solver_name in keys(matched_solvers)
            for size_symbol in [:small_time, :med_time, :large_time, :total_time]
                matched_solvers[solver_name][size_symbol] = matched_solvers[solver_name][size_symbol]^(1/n_dict[size_symbol]) - delta
            end
            for size_symbol in [:small_cvio, :med_cvio, :large_cvio, :total_cvio]
                matched_solvers[solver_name][size_symbol] = matched_solvers[solver_name][size_symbol]^(1/matched_solvers[solver_name][Symbol(string(size_symbol)[1:end-4]*"count")]) - delta
            end

            push!(df, (
                tol, solver_name,
                matched_solvers[solver_name][:small_count], matched_solvers[solver_name][:small_time], matched_solvers[solver_name][:small_cvio],
                matched_solvers[solver_name][:med_count], matched_solvers[solver_name][:med_time], matched_solvers[solver_name][:med_cvio],
                matched_solvers[solver_name][:large_count], matched_solvers[solver_name][:large_time], matched_solvers[solver_name][:large_cvio],
                matched_solvers[solver_name][:total_count], matched_solvers[solver_name][:total_time], matched_solvers[solver_name][:total_cvio]
            ))

        end
        
    end
    
    CSV.write(save_prefix * ".csv", df)
    open(save_prefix * ".txt", "w") do io
        show(io, df, allrows=true, allcols=true)
    end

    # ----------------
    # Build LaTeX table
    # ----------------
    

    header = """
\\begin{center}
\\renewcommand{\\arraystretch}{0.9}
\\begin{tabular}{|l|l|ccc|ccc|ccc|ccc|}
\\hline
 & & \\multicolumn{3}{c|}{\\textbf{Small ($(n_small))}} & \\multicolumn{3}{c|}{\\textbf{Medium ($(n_med))}} & \\multicolumn{3}{c|}{\\textbf{Large ($(n_large))}} & \\multicolumn{3}{c|}{\\textbf{Total ($(n_total))}} \\\\
 & & \\textbf{Count} & \\textbf{Time} & \\textbf{Cvio} & \\textbf{Count} & \\textbf{Time} & \\textbf{Cvio} & \\textbf{Count} & \\textbf{Time} & \\textbf{Cvio} & \\textbf{Count} & \\textbf{Time} & \\textbf{Cvio}\\\\
\\hline
"""
 
    solver_order = [
        "MadNLP+LiftedKKT (GPU)",
        "MadNLP+HybridKKT (GPU)",
        "MadNCL (GPU)",
        "Ipopt+Ma27 (CPU)",
        "MadNLP+Ma86 (CPU)"
    ]

    body = IOBuffer()
    for tol in unique(df.tol)
        tol_rows = filter(row -> row.tol == tol, df)
        exp = Int(round(log10(1/tol)))
        tol_str = "\\Large\\textbf{\$10^{-$exp}\$}"

        # reorder rows by solver_order
        rows = collect(eachrow(tol_rows))  # turn into iterable rows
        sorted_rows = []
        for s in solver_order
            idx = findfirst(r -> r.solver == s, rows)
            if idx !== nothing
                push!(sorted_rows, rows[idx])
            end
        end
        sorted_rows = filter(!isnothing, sorted_rows)

        # collect values for highlighting
        counts = Dict(
            :small_count => maximum([r.small_count for r in sorted_rows]),
            :med_count   => maximum([r.med_count   for r in sorted_rows]),
            :large_count => maximum([r.large_count for r in sorted_rows]),
            :total_count => maximum([r.total_count for r in sorted_rows])
        )
        times = Dict(
            :small_time => minimum([r.small_time for r in sorted_rows]),
            :med_time   => minimum([r.med_time   for r in sorted_rows]),
            :large_time => minimum([r.large_time for r in sorted_rows]),
            :total_time => minimum([r.total_time for r in sorted_rows])
        )

        cvios = Dict(
            :small_cvio => minimum([r.small_cvio for r in sorted_rows]),
            :med_cvio   => minimum([r.med_cvio   for r in sorted_rows]),
            :large_cvio => minimum([r.large_cvio for r in sorted_rows]),
            :total_cvio => minimum([r.total_cvio for r in sorted_rows])
        )

        firstrow = true
        for r in sorted_rows
            # helper to maybe highlight
            function fmt(val, key; is_time=false, is_cvio=false, times=nothing, counts=nothing, cvios=nothing)
                if is_time
                    return val == times[key] ? "\\cellcolor{blue!15}$(round(val, digits=2))" : "$(round(val,digits=2))"
                elseif is_cvio
                    sval = @sprintf("%.3e", val)
                    return val == cvios[key] ? "\\cellcolor{blue!15}$(sval)" : "$(sval)"
                else
                    return val == counts[key] ? "\\cellcolor{blue!15}$(val)" : "$(val)"
                end
            end

            line = ""
            if firstrow
                line *= "\\multirow{$(length(sorted_rows))}{*}{$tol_str} & "
                firstrow = false
            else
                line *= " & "
            end

            #=line *= "\\textbf{$(r.solver)} & " *
                    fmt(r.small_count,:small_count) * " & " * fmt(r.small_time,:small_time,is_time=true) * " & " *
                    fmt(r.med_count,:med_count)     * " & " * fmt(r.med_time,:med_time,is_time=true)     * " & " *
                    fmt(r.large_count,:large_count) * " & " * fmt(r.large_time,:large_time,is_time=true) * " & " *
                    fmt(r.total_count,:total_count) * " & " * fmt(r.total_time,:total_time,is_time=true) * " \\\\"=#

            line *= "\\textbf{$(r.solver)} & " *
                fmt(r.small_count,:small_count; counts=counts) * " & " * fmt(r.small_time,:small_time,is_time=true; times=times) * " & " * fmt(r.small_cvio,:small_cvio,is_cvio=true,cvios=cvios) * " & " *
                fmt(r.med_count,:med_count; counts=counts)     * " & " * fmt(r.med_time,:med_time,is_time=true; times=times)     * " & " * fmt(r.med_cvio,:med_cvio,is_cvio=true,cvios=cvios) * " & " *
                fmt(r.large_count,:large_count; counts=counts) * " & " * fmt(r.large_time,:large_time,is_time=true; times=times) * " & " * fmt(r.large_cvio,:large_cvio,is_cvio=true,cvios=cvios) * " & " *
                fmt(r.total_count,:total_count; counts=counts) * " & " * fmt(r.total_time,:total_time,is_time=true; times=times) * " & " * fmt(r.total_cvio,:total_cvio,is_cvio=true,cvios=cvios) * " \\\\"


            println(body, line)
        end
        println(body, "\\hline")
    end

    footer = """
\\end{tabular}
\\end{center}
"""

    open(save_prefix * ".tex", "w") do io
        write(io, header * String(take!(body)) * footer)
    end
end


            
             

function solve_benchmark_cases(cases, tol, hardware; coords = "Polar", case_style = "default", curve = [.64, .60, .58, .56, .56, .58, .64, .76, .87, .95, .99, 1.0, .99, 1.0, 1.0,
    .97, .96, .96, .93, .92, .92, .93, .87, .72, .64], mp = false, storage = false, sc = false, corrective_action_ratio = 0.25, include_ctg = true)

    max_wall_time = Float64(900)
    max_iter = Int64(3000)
    if sc
        max_wall_time = Float64(30000)
        max_iter = Int(10000)
    end

    if storage 
        csv_filename = "saved_raw_data/benchmark_results_mpopf_stor_" *hardware *"_" * case_style * "_tol_" * replace(@sprintf("%.0e", 1 / tol), r"\+0?" => "") * "_" * coords * ".csv"
    elseif mp
        csv_filename = "saved_raw_data/benchmark_results_mpopf_" *hardware *"_" * case_style * "_tol_" * replace(@sprintf("%.0e", 1 / tol), r"\+0?" => "") * "_" * coords * ".csv"
    elseif sc
        csv_filename = "saved_raw_data/benchmark_results_scopf_" *hardware *"_" * case_style * "_tol_" * replace(@sprintf("%.0e", 1 / tol), r"\+0?" => "") *".csv"
    else
        csv_filename = "saved_raw_data/benchmark_results_opf_" *hardware *"_" * case_style * "_tol_" * replace(@sprintf("%.0e", 1 / tol), r"\+0?" => "") * "_" * coords * ".csv"
    end

    if coords == "Polar"
        form = :polar
    elseif coords == "Rectangular"
        form = :rect
    else
        error("Wrong coords")
    end

    schema = Dict(
        :id          => Int,
        :iter        => Any,
        :soltime     => Float64,
        :inittime    => Any,
        :adtime      => Any,
        :lintime     => Any,
        :termination => String,
        :obj         => Any,
        :cvio        => Any
    )

    function enforce_schema!(df, schema::Dict{Symbol,DataType})
        for (col, T) in schema
            if string(col) in names(df)
                df[!, col] = Vector{T}(df[!, col])
            else
                df[!, col] = Vector{T}()  # add missing col
            end
        end
        return df
    end

    existing_results = Dict{Symbol,DataFrame}()
    if isfile(csv_filename)
        file_exists = true
        println("Found existing results at $csv_filename")
        existing_results = CSV.read(csv_filename, DataFrame)
    else
        println("No existing results found")
        file_exists = false
    end


    

    if !file_exists
        df_top = DataFrame(
            nvar = Int[],
            ncon = Int[],
            case_name = String[]
        )
    else
        df_top = select(existing_results, r"_top$")
        rename!(df_top, Symbol.(replace.(String.(names(df_top)), "_top" => "")))
    end

    #Compile time on smallest case
    if hardware == "GPU"
        if storage
            model_gpu, ~ = mpopf_model("pglib_opf_case3_lmbd_storage", curve; backend = CUDABackend(), form=form, corrective_action_ratio = corrective_action_ratio)
        elseif mp
            model_gpu, ~ = mpopf_model("pglib_opf_case3_lmbd", curve; backend = CUDABackend(), form=form, corrective_action_ratio = corrective_action_ratio)
        elseif sc
            test_case = "data/C3E4N00073D1_scenario_303.json"
            test_uc_case = "data/C3E4N00073D1_scenario_303_solution.json"
            model_gpu, ~ = scopf_model(test_case, test_uc_case; backend = CUDABackend(), include_ctg = include_ctg)
        else
            model_gpu, ~ = opf_model("pglib_opf_case3_lmbd"; backend = CUDABackend(), form=form)
        end
        ~ = madnlp(model_gpu, tol = tol, max_iter = 3, disable_garbage_collector=false, dual_initialized=true)
        ~ = MadNCL.madncl(model_gpu, tol = tol, max_iter = 3, disable_garbage_collector=false, dual_initialized=true, 
                        kkt_system=MadNCL.K2rAuglagKKTSystem,
                        linear_solver=MadNLPGPU.CUDSSSolver,
                        cudss_pivot_epsilon=1e-12,
                        ncl_options = MadNCL.NCLOptions{Float64}(
                            scaling_max_gradient=100,
                            feas_tol=tol,
                            opt_tol=tol,
                            scaling=true,
                        ))
        ~ = madnlp(model_gpu, tol=tol, max_wall_time = max_wall_time, disable_garbage_collector=false, dual_initialized=true, linear_solver=MadNLPGPU.CUDSSSolver,
                                            cudss_algorithm=MadNLP.LDL,
                                            kkt_system=HybridKKT.HybridCondensedKKTSystem,
                                            equality_treatment=MadNLP.EnforceEquality,
                                            fixed_variable_treatment=MadNLP.MakeParameter, max_iter = 3)
        
        if !file_exists
            df_lifted_kkt = DataFrame(
                id = Int[], iter = Any[], soltime = Float64[], inittime = Any[],
                adtime = Any[], lintime = Any[], termination = String[],
                obj = Any[], cvio = Any[]
            )
            df_hybrid_kkt = similar(df_lifted_kkt)
            df_madncl = similar(df_lifted_kkt)
        else
            df_lifted_kkt = select(existing_results, r"lifted_kkt$")
            rename!(df_lifted_kkt, Symbol.(replace.(String.(names(df_lifted_kkt)), "_lifted_kkt" => "")))
            df_lifted_kkt = enforce_schema!(df_lifted_kkt, schema)
            df_hybrid_kkt = select(existing_results, r"hybrid_kkt$")
            rename!(df_hybrid_kkt, Symbol.(replace.(String.(names(df_hybrid_kkt)), "_hybrid_kkt" => "")))
            df_hybrid_kkt = enforce_schema!(df_hybrid_kkt, schema)
            df_madncl = select(existing_results, r"madncl$")
            rename!(df_madncl, Symbol.(replace.(String.(names(df_madncl)), "_madncl" => "")))
            df_madncl = enforce_schema!(df_madncl, schema)

        end
        
        
    elseif hardware == "CPU"
        if storage
            model_cpu, ~ = mpopf_model("pglib_opf_case3_lmbd_storage", curve; form=form, corrective_action_ratio = corrective_action_ratio)
        elseif mp
            model_cpu, ~ = mpopf_model("pglib_opf_case3_lmbd", curve; form=form, corrective_action_ratio = corrective_action_ratio)
        elseif sc
            test_case = "data/C3E4N00073D1_scenario_303.json"
            test_uc_case = "data/C3E4N00073D1_scenario_303_solution.json"
            model_cpu, ~ = scopf_model(test_case, test_uc_case; include_ctg = include_ctg)
        else
            model_cpu, ~ = opf_model("pglib_opf_case3_lmbd"; form=form)
        end
        ~ = ipopt(model_cpu, tol = tol, max_iter = 3, dual_inf_tol=Float64(10000), constr_viol_tol=Float64(10000), compl_inf_tol=Float64(10000), linear_solver = "ma27")
        #~ = ipopt(model_cpu, tol = tol, max_iter = 3, dual_inf_tol=Float64(10000), constr_viol_tol=Float64(10000), compl_inf_tol=Float64(10000), bound_relax_factor = tol, linear_solver = "ma97")
        ~ = madnlp(model_cpu, tol = tol,
                kkt_system=MadNLP.SparseCondensedKKTSystem, equality_treatment=MadNLP.RelaxEquality, 
                fixed_variable_treatment=MadNLP.RelaxBound, dual_initialized=true,
                linear_solver=MadNLPHSL.Ma86Solver, ma86_num_threads=28, max_iter = 3)

        if !file_exists
            df_ma27 = DataFrame(
                id = Int[], iter = Int[], soltime = Float64[], adtime = Float64[],
                termination = String[], obj = Float64[], cvio = Float64[]
            )
            df_ma86 = DataFrame(
                id = Int[], iter = Float64[], soltime = Float64[], inittime = Float64[],
                adtime = Float64[], lintime = Float64[], termination = String[],
                obj = Float64[], cvio = Float64[]
            )
            #df_ma97 = similar(df_ma27)
        else
            df_ma27 = select(existing_results, r"ma27$")
            rename!(df_ma27, Symbol.(replace.(String.(names(df_ma27)), "_ma27" => "")))
            df_ma86 = select(existing_results, r"ma86$")
            rename!(df_ma86, Symbol.(replace.(String.(names(df_ma86)), "_ma86" => "")))
        end
    else
        error("Invalid hardware input")
    end

    existing_cases = Set(df_top.case_name)
    println("Already have $(length(existing_cases)) cases stored.")
    

    for (i, case) in enumerate(cases)
        println(case)

        if sc
            (problem_case, uc_case) = case
            case = replace(problem_case, r"^data/|\.json$" => "")
        elseif !storage
            if case_style == "default"
                case = case*".m"
            elseif case_style == "api"
                case = "api/"*case*"__api.m"
            elseif case_style == "sad"
                case = "sad/"*case*"__sad.m"
            else
                error("Invalid case style")
            end
        else
            if case_style == "default"
                case = case*"_storage"
            elseif case_style == "api"
                case = case*"__api_storage"
            elseif case_style == "sad"
                case = case*"__sad_storage"
            else
                error("Invalid case style")
            end
        end

        if case in existing_cases
            println("Skipping $case (already in results)")
            continue
        end

        if hardware == "GPU"
            #GPU 
            let
                if storage || mp
                    m_gpu, v_gpu, c_gpu = mpopf_model(case, curve; backend = CUDABackend(), form=form, corrective_action_ratio = corrective_action_ratio)  
                elseif sc
                    m_gpu, v_gpu, c_gpu = scopf_model(problem_case, uc_case; backend = CUDABackend(), include_ctg = include_ctg)   
                else 
                    m_gpu, v_gpu, c_gpu = opf_model(case; backend = CUDABackend(), form=form)
                end   
                push!(df_top, (m_gpu.meta.nvar, m_gpu.meta.ncon, case))

                try
                    result_lifted_kkt = madnlp(m_gpu, tol=tol, max_wall_time = max_wall_time, disable_garbage_collector=false, dual_initialized=true, max_iter=max_iter)
                    c = evaluate(m_gpu, result_lifted_kkt)
                    push!(df_lifted_kkt, (i, result_lifted_kkt.counters.k, result_lifted_kkt.counters.total_time, result_lifted_kkt.counters.init_time, result_lifted_kkt.counters.eval_function_time, 
                    result_lifted_kkt.counters.linear_solver_time, termination_code(result_lifted_kkt.status), result_lifted_kkt.objective, c))
                catch e
                    if occursin("Out of GPU memory", sprint(showerror, e))
                        @warn "GPU OOM on this problem, skipping..."
                        push!(df_lifted_kkt, (i, "-", max_wall_time, "-", "-", "-", "me", "-", "-"))
                    else
                        rethrow(e)
                    end
                end
            end
            GC.gc()
            CUDA.reclaim()
            println("lifted")
            CUDA.memory_status()
                

            let 
                if storage || mp
                    m_gpu, v_gpu, c_gpu = mpopf_model(case, curve; backend = CUDABackend(), form=form, corrective_action_ratio = corrective_action_ratio)  
                elseif sc
                    m_gpu, v_gpu, c_gpu = scopf_model(problem_case, uc_case; backend = CUDABackend(), include_ctg = include_ctg)   
                else 
                    m_gpu, v_gpu, c_gpu = opf_model(case; backend = CUDABackend(), form=form)
                end
                try

                    solver = MadNLP.MadNLPSolver(m_gpu; tol=tol, max_wall_time = max_wall_time, disable_garbage_collector=false, dual_initialized=true, linear_solver=MadNLPGPU.CUDSSSolver,
                                                cudss_algorithm=MadNLP.LDL,
                                                kkt_system=HybridKKT.HybridCondensedKKTSystem,
                                                equality_treatment=MadNLP.EnforceEquality,
                                                fixed_variable_treatment=MadNLP.MakeParameter,
                                                max_iter=max_iter)
                    solver.kkt.gamma[] = 1e7
                    result_hybrid_kkt = MadNLP.solve!(solver)

                    c = evaluate(m_gpu, result_hybrid_kkt)
                    push!(df_hybrid_kkt, (i, result_hybrid_kkt.counters.k, result_hybrid_kkt.counters.total_time, result_hybrid_kkt.counters.init_time, result_hybrid_kkt.counters.eval_function_time, 
                    result_hybrid_kkt.counters.linear_solver_time, termination_code(result_hybrid_kkt.status), result_hybrid_kkt.objective, c))
                catch e
                    if occursin("Out of GPU memory", sprint(showerror, e))
                        @warn "GPU OOM on this problem, skipping..."
                        push!(df_hybrid_kkt, (i, "-", max_wall_time, "-", "-", "-", "me", "-", "-"))
                    else
                        rethrow(e)
                    end
                end

                
            end
            GC.gc()
            CUDA.reclaim()
            println("hybrid")
            CUDA.memory_status()

            let
                if storage || mp
                    m_gpu, v_gpu, c_gpu = mpopf_model(case, curve; backend = CUDABackend(), form=form, corrective_action_ratio = corrective_action_ratio)  
                elseif sc
                    m_gpu, v_gpu, c_gpu = scopf_model(problem_case, uc_case; backend = CUDABackend(), include_ctg = include_ctg)   
                else 
                    m_gpu, v_gpu, c_gpu = opf_model(case; backend = CUDABackend(), form=form)
                end

                try
                    result_madncl = MadNCL.madncl(m_gpu, tol=tol, max_wall_time = max_wall_time, disable_garbage_collector=false, 
                                                    dual_initialized=true, 
                                                    kkt_system=MadNCL.K2rAuglagKKTSystem,
                                                    linear_solver=MadNLPGPU.CUDSSSolver,
                                                    cudss_pivot_epsilon=1e-12,
                                                    ncl_options = MadNCL.NCLOptions{Float64}(scaling_max_gradient=100, feas_tol=tol, opt_tol=tol, scaling=true),
                                                    max_iter=max_iter)
                    c = evaluate(m_gpu, result_madncl)
                    push!(df_madncl, (i, result_madncl.counters.k, result_madncl.counters.total_time, result_madncl.counters.init_time, result_madncl.counters.eval_function_time, 
                    result_madncl.counters.linear_solver_time, termination_code(result_madncl.status), result_madncl.objective, c))
                catch e
                    if occursin("Out of GPU memory", sprint(showerror, e))
                        @warn "GPU OOM on this problem, skipping..."
                        push!(df_madncl, (i, "-", max_wall_time, "-", "-", "-", "me", "-", "-"))
                    else
                        rethrow(e)
                    end
                end
                result_madncl = nothing
                m_gpu = nothing
                v_gpu = nothing
                c_gpu = nothing
            end
            GC.gc()
            CUDA.reclaim()
            println("madncl")
            CUDA.memory_status()

            opf_results = Dict(:top => df_top,
            :lifted_kkt => df_lifted_kkt,
            :hybrid_kkt => df_hybrid_kkt,
            :madncl => df_madncl,)
        
        elseif hardware == "CPU"
            #CPU
            if storage || mp
                m_cpu, v_cpu, c_cpu = mpopf_model(case, curve; form=form, corrective_action_ratio = corrective_action_ratio)  
            elseif sc
                m_cpu, v_cpu, c_cpu = scopf_model(problem_case, uc_case; include_ctg = include_ctg)
            else 
                m_cpu, v_cpu, c_cpu = opf_model(case; form=form)
            end
            push!(df_top, (m_cpu.meta.nvar, m_cpu.meta.ncon, case))

            result_ma27 = ipopt(m_cpu, tol = tol, max_wall_time=max_wall_time, dual_inf_tol=Float64(10000), 
            constr_viol_tol=Float64(10000), compl_inf_tol=Float64(10000), linear_solver = "ma27",
             honor_original_bounds = "no", print_timing_statistics = "yes", output_file = "ipopt_output", max_iter=max_iter)
            it, tot, ad = ipopt_stats("ipopt_output")
            c = evaluate(m_cpu, result_ma27)
            push!(df_ma27, (i, it, tot, ad, termination_code(result_ma27.solver_specific[:internal_msg]), result_ma27.objective, c))

            result_ma86 = madnlp(m_cpu, tol = tol, max_wall_time=max_wall_time,
                kkt_system=MadNLP.SparseCondensedKKTSystem, equality_treatment=MadNLP.RelaxEquality, 
                fixed_variable_treatment=MadNLP.RelaxBound, dual_initialized=true,
                linear_solver=MadNLPHSL.Ma86Solver, ma86_num_threads=28, max_iter=max_iter)
            c = evaluate(m_cpu, result_ma86)
            push!(df_ma86, (i, result_ma86.counters.k, result_ma86.counters.total_time, result_ma86.counters.init_time, result_ma86.counters.eval_function_time, 
                result_ma86.counters.linear_solver_time, termination_code(result_ma86.status), result_ma86.objective, c))

            #=result_ma97 = ipopt(m_cpu, tol = tol, max_wall_time=max_wall_time, dual_inf_tol=Float64(10000), constr_viol_tol=Float64(10000), compl_inf_tol=Float64(10000), bound_relax_factor = tol, linear_solver = "ma97", honor_original_bounds = "no", print_timing_statistics = "yes", output_file = "ipopt_output")
            it, tot, ad = ipopt_stats("ipopt_output")
            c = evaluate(m_cpu, result_ma97)
            push!(df_ma97, (i, it, tot, ad, termination_code(result_ma97.solver_specific[:internal_msg]), result_ma97.objective, c))=#

            opf_results = Dict(:top => df_top,
                :ma27 => df_ma27,
                :ma86 => df_ma86,)
                #:ma97 => df_ma97)
        end

        save_opf_results(opf_results, csv_filename)
    end

end    
    
    #generate_tex_opf(opf_results, coords; filename = "select_saved_data/benchmark_results_opf_" * case_style * "_tol_" * replace(@sprintf("%.0e", 1 / tol), r"\+0?" => "")*"_"*coords*".tex")

    #return opf_results
#end

curves = Dict("easy" => [.64, .60, .58, .56, .56, .58, .64, .76, .87, .95, .99, 1.0, .99, 1.0, 1.0,
    .97, .96, .96, .93, .92, .92, .93, .87, .72, .64],)
    #"medium" => [.88, .90, .88, .86, .87, .88, .9, .92, .93, .95, .97, 1.0, .99, 1.0, 1.0,
    #.97, .96, .96, .93, .92, .92, .93, .89, .85, .82],
    #"hard" => [.52, .60, .53, .59, .51, .62, .65, .76, .87, .95, .99, 1.01, .99, 1.0, 1.02,
    #.92, 1.0, .9, .93, .84, .92, .93, .85, .73, .62])



sc_cases = [("data/C3E4N00073D1_scenario_303.json", "data/C3E4N00073D1_scenario_303_solution.json"),
("data/C3E4N00073D2_scenario_303.json", "data/C3E4N00073D2_scenario_303_solution.json"),
 ("data/C3E4N00073D3_scenario_303.json", "data/C3E4N00073D3_scenario_303_solution.json"),
("data/C3E4N00073D2_scenario_911.json", "data/C3E4N00073D2_scenario_911_solution.json"), 
("data/C3E4N00617D1_scenario_002.json", "data/C3E4N00617D1_scenario_002_solution.json"),
("data/C3E4N00617D2_scenario_002.json", "data/C3E4N00617D2_scenario_002_solution.json"),
("data/C3E4N00617D3_scenario_002.json", "data/C3E4N00617D3_scenario_002_solution.json"),
("data/C3E4N00617D1_scenario_921.json", "data/C3E4N00617D1_scenario_921_solution.json"),]

function eval_model_build(cases; coords = "Polar")
    if coords == "Polar"
        form = :polar
    elseif coords == "Rectangular"
        form = :rect
    else
        error("Invalid coords")
    end

    #compile
    filename = "data/pglib_opf_case3_lmbd.m"
    #PowerModels
    data_pm = PowerModels.parse_file(filename)
    m_pm = JuMP.Model()
    pm = instantiate_model(data_pm, ACPPowerModel, PowerModels.build_opf, jump_model = m_pm)
    nlp_pm = MathOptNLPModel(m_pm)

    #ExaModelsPower
    #CPU
    m, v, c = opf_model(filename;)
    #GPU
    m, v, c = opf_model(filename; backend = CUDABackend())

    results = DataFrame(
        case = String[],
        method = String[],
        time_sec = Float64[],
    )

    for case in cases
        case = "data/"*case*".m"

        ## --- PowerModels ---
        t1 = time()
        data_pm = PowerModels.parse_file(filename)
        m_pm = JuMP.Model()
        pm = instantiate_model(data_pm, ACPPowerModel, PowerModels.build_opf, jump_model = m_pm)
        nlp_pm = MathOptNLPModel(m_pm)
        t_pm = time() - t1
        push!(results, (case, "PowerModels-CPU", t_pm))

        ## --- ExaModels CPU ---
        t1 = time()
        m, v, c = opf_model(filename;)
        t_cpu = time() - t1
        push!(results, (case, "ExaModels-CPU", t_cpu))

        ## --- ExaModels GPU ---
        t1 = time()
        m, v, c = opf_model(filename; backend = CUDABackend())
        t_gpu = time() - t1
        push!(results, (case, "ExaModels-GPU", t_gpu))

    end

    csv_file = "model_build_benchmark/"*coords*".csv"
    txt_file = "model_build_benchmark/"*coords*".txt"

    CSV.write(csv_file, results)

    open(txt_file, "w") do io
        show(io, MIME("text/plain"), results)
    end


    delta = 10

    time_df = Dict("PowerModels-CPU" => Float64(1), "ExaModels-CPU" => Float64(1), "ExaModels-GPU" => Float64(1))
    count_df = Dict("PowerModels-CPU" => 0, "ExaModels-CPU" => 0, "ExaModels-GPU" => 0)
    for result in eachrow(results)
        count_df[result[:method]] += 1
        time_df[result[:method]] = time_df[result[:method]]*(result[:time_sec]+delta)
    end

    summary = DataFrame(platform = Any[], Time = Any[])

    for solver in ("PowerModels-CPU", "ExaModels-CPU", "ExaModels-GPU")
        mean_time = time_df[solver]^(1/count_df[solver]) - delta
        push!(summary, (solver, mean_time))
    end


    # Save summary as CSV and TXT
    summary_csv = "model_build_benchmark/summary_"*coords*".csv"
    summary_txt = "model_build_benchmark/summary_"*coords*".txt"
    summary_tex = "model_build_benchmark/summary_"*coords*".tex"

    CSV.write(summary_csv, summary)
    open(summary_txt, "w") do io
        show(io, MIME("text/plain"), summary)
    end

    # Save summary as LaTeX table
    header = """
\\begin{center}
\\renewcommand{\\arraystretch}{0.9}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Platform} & \\textbf{Time (s)} \\\\
\\hline
"""
    body = IOBuffer()
    for row in eachrow(summary)
        println(body, (row.platform)*" & \$"*string((round(row.Time, digits=4)))*" \\\\")
    end
    footer = """
\\hline
\\end{tabular}
\\end{center}
"""
    open(summary_tex, "w") do io
        write(io, header * String(take!(body)) * footer)
    end

end

