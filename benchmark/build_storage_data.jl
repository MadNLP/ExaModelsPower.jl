function modify_case_with_storage(input_file::String, output_file::String)
    lines = readlines(input_file)
    
    # Insert header comment above function definition
    func_idx = findfirst(l -> occursin("function mpc =", l), lines)
    insert!(lines, func_idx, "% Modified to have additional storage capacity and increased gencost")

    # Extract mpc.bus data
    bus_start = findfirst(contains("mpc.bus = ["), lines)
    bus_end = findnext(contains("];"), lines, bus_start)
    bus_lines = lines[bus_start+1 : bus_end-1]
    bus_data = [
        parse.(Float64, split(strip(replace(l, ";" => ""))))
        for l in bus_lines if !isempty(strip(l))
    ]
    # Get top 10% Pd (col 3)
    pd_values = [row[3] for row in bus_data]
    pd_positive = filter(x -> x[2] > 0, collect(zip(1:length(pd_values), pd_values)))
    sorted_pd = sort(pd_positive, by = x -> -x[2])
    n_top = max(1, round(Int, 0.1 * length(bus_data)))
    top_indices = first.(sorted_pd[1:n_top])

    # Prepare mpc.storage section
    storage_header = [
        "%% storage data",
        "%	bus	energy	energy_rating	charge_rating	discharge_rating	thermal_rating	qmin	qmax	r	x	standby_loss	status",
        "mpc.storage = ["
    ]
    storage_entries = String[]
    for i in top_indices
        bus_id = Int(bus_data[i][1])
        pd = bus_data[i][3]
        push!(storage_entries,
            @sprintf("\t%d\t0.0\t0.0\t1.00\t%.4f\t%.4f\t%.4f\t0.9\t0.85\t1000\t-1000\t1000\t0.1\t0.01\t0\t0\t1;",
                     bus_id, 1.5*pd, 0.75*pd, 0.54*pd))
    end
    push!(storage_entries, "];")

    #Ratio of energy rating to peak bus output is set at 1.5, loosely based on CAISO's ratio of 1.27 for the entire grid
    #https://www.caiso.com/documents/gross-and-net-load-peaks-fact-sheet.pdf
    #https://www.caiso.com/documents/2024-special-report-on-battery-storage-may-29-2025.pdf
    #Ratio of energy capacity to charge and discharge rating, as well as all other storage values from values used in test case
    #https://arxiv.org/pdf/2004.14768

    # Modify mpc.gencost lines
    gen_start = findfirst(contains("mpc.gencost = ["), lines)
    gen_end = findnext(contains("];"), lines, gen_start)
    for i in gen_start+1:gen_end-1
        fields = split(strip(lines[i]))
        if length(fields) >= 6
            vg = parse(Float64, fields[6])
            c1 = parse(Float64, fields[5])
            if vg != 0.0 && c1 == 0.0
                fields[5] = "0.2"
                lines[i] = join(fields, '\t')
            end
        end
    end

    # Insert storage section after gen_end
    insert_at = gen_end + 1
    insert!(lines, insert_at, "")  # blank line
    for i in reverse(storage_header)
        insert!(lines, insert_at+1, i)
    end
    for i in reverse(storage_entries)
        insert!(lines, insert_at + 1 + length(storage_header), i)
    end

    # Write to output
    open(output_file, "w") do f
        for line in lines
            println(f, line)
        end
    end
end
