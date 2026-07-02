import ExaPowerIO
import PowerIO

_case_path(name) = joinpath(@__DIR__, "..", "data", name)

_matches(lhs::Real, rhs::Real) = isapprox(lhs, rhs; rtol = 1e-9, atol = 1e-10, nans = true)
_matches(lhs, rhs) = lhs == rhs

function _test_matching_rows(actual, expected)
    @test length(actual) == length(expected)
    for (a, e) in zip(actual, expected), field in propertynames(e)
        @test _matches(getproperty(a, field), getproperty(e, field))
    end
end

function _test_matches_exapowerio(actual, expected)
    @test actual.baseMVA == [expected.baseMVA]
    for table in (:bus, :gen, :branch, :arc, :storage)
        _test_matching_rows(getproperty(actual, table), getproperty(expected, table))
    end
end

@testset "PowerIO parser backend" begin
    matpower_cases = [
        "pglib_opf_case3_lmbd.m",
        "pglib_opf_case5_pjm.m",
        "pglib_opf_case14_ieee.m",
        "pglib_opf_case3_lmbd_mod.m",
        "pglib_opf_case5_pjm_mod.m",
    ]

    for case in matpower_cases
        path = _case_path(case)
        expected = ExaPowerIO.parse_matpower(Float64, path)

        default_actual = ExaModelsPower.parse_ac_power_data(path, Float64)
        _test_matches_exapowerio(default_actual, expected)

        exapowerio_actual =
            ExaModelsPower.parse_ac_power_data(path, Float64; parser = :exapowerio)
        _test_matches_exapowerio(exapowerio_actual, expected)
    end

    @testset "GO Challenge 3 input parsing comes from PowerIO" begin
        path = _case_path("C3E4N00073D1_scenario_303.json")
        data = PowerIO.parse_goc3_json(path)
        raw = data.raw

        @test haskey(raw, "network")
        @test !isempty(data.bus_lookup)
        @test !isempty(data.sdd_lookup)
        @test !isempty(data.sdd_ts_lookup)
        @test data.bus_ids == sort(data.bus_ids)
        @test all(uid -> data.bus_id_by_uid[uid] isa Integer, data.bus_ids)
        @test data.sdd_ids == sort(data.sdd_ids)
        @test haskey(data.violation_cost, "p_bus_vio_cost")
        @test length(data.periods) == length(data.dt)

        rows = [Dict("uid" => first(data.sdd_ids), "on_status" => [1, 0, 1])]
        PowerIO.goc3_add_status_flags!(rows, data.sdd_lookup)
        @test haskey(rows[1], "su_status")
        @test haskey(rows[1], "sd_status")
    end

    mktempdir() do dir
        matpower = _case_path("pglib_opf_case3_lmbd.m")
        psse_path = joinpath(dir, "case3.raw")
        powerworld_path = joinpath(dir, "case3.aux")
        powermodels_path = joinpath(dir, "case3.json")

        write(psse_path, first(PowerIO.convert_file(matpower, "psse")))
        write(powerworld_path, first(PowerIO.convert_file(matpower, "powerworld")))
        write(powermodels_path, first(PowerIO.convert_file(matpower, "powermodels-json")))

        for path in (psse_path, powerworld_path)
            data = ExaModelsPower.parse_ac_power_data(path, Float64)
            @test length(data.bus) == 3
            @test length(data.branch) == 3
            @test length(data.gen) == 3
        end

        pm_data = ExaModelsPower.parse_ac_power_data(
            powermodels_path,
            Float64;
            from = "powermodels",
        )
        @test length(pm_data.bus) == 3
        @test length(pm_data.branch) == 3
        @test length(pm_data.gen) == 3

        @test_throws ErrorException ExaModelsPower.parse_ac_power_data(
            powermodels_path,
            Float64;
            parser = :exapowerio,
            from = "powermodels",
        )

        @test !isnothing(ac_opf_model(psse_path; form = :polar)[1])
        @test !isnothing(ac_opf_model(powerworld_path; form = :rect)[1])
        @test !isnothing(dcopf_model(psse_path)[1])
        @test !isnothing(
            mpopf_model(
                powermodels_path,
                [1.0, 0.95];
                from = "powermodels",
            )[1],
        )
    end
end
