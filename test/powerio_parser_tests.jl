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

# Backend-free: everything here runs on the serial path, so `runtests` calls it from
# one slice rather than letting it repeat in every CI job.
function powerio_parser_tests()
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
            actual = ExaModelsPower.parse_ac_power_data(path, Float64)
            _test_matches_exapowerio(actual, expected)
        end

        # Only bare names use the PGLib fallback. An explicit path must reach
        # PowerIO unchanged even when it is missing, so its error names the
        # caller's path rather than a fabricated path inside the artifact.
        let missing = joinpath("missing-cases", "case.m")
            @test ExaModelsPower._case_path(missing) == missing
            @test isfile(ExaModelsPower._case_path("pglib_opf_case3_lmbd.m"))
        end

        # Every model here is built from the parse, and a single abstract field in it
        # leaves `ExaModel(CORE, data)` unresolved under `--trim=safe`. The row types
        # must also be isbits, or the GPU backends refuse the arrays built from them.
        #
        # Both halves hold as of PowerIO 0.9: the library pins its return type, and the
        # wrapper forwards `T` positionally so the static parameter survives the call.
        # Either one regressing widens the rows back to an abstract element type.
        let path = _case_path("pglib_opf_case14_ieee.m")
            @test isconcretetype(
                Base.infer_return_type(PowerIO.parse_ac_power_data, (String,)),
            )
            @test isconcretetype(
                Base.infer_return_type(ExaModelsPower.parse_ac_power_data, (String, Type{Float64})),
            )
            p = ExaModelsPower.parse_ac_power_data(path, Float64)
            for table in (:bus, :gen, :branch, :arc)
                @test isbitstype(eltype(getproperty(p, table)))
            end
        end

        # The multiperiod busarray is built immutably from the LoadSeries; its element
        # type must stay concrete/isbits so the GPU kernels accept it (mirrors the
        # SCOPF boundary test, and surfaces a PowerIO regression without a GPU runner).
        let mp_net = ExaModelsPower.parse_mp_network(_case_path("pglib_opf_case3_lmbd.m"))
            mp_series = PowerIO.LoadSeries(mp_net, [1.0, 0.95], Float64)
            mp_data = ExaModelsPower.parse_mp_power_data(mp_net, mp_series, 2, 0.1, Float64)
            @test isconcretetype(eltype(mp_data.busarray))
            @test isbitstype(eltype(mp_data.busarray))

            # A horizon longer than the series is a clear error rather than a
            # BoundsError from inside the comprehension.
            @test_throws DimensionMismatch ExaModelsPower.parse_mp_power_data(
                mp_net, mp_series, 3, 0.1, Float64)
        end

        # The exact multiperiod argument type is part of the compiled library
        # boundary. Concrete rows at run time are not enough: juliac needs to
        # infer the same closed type before it emits the library.
        let path = _case_path("pglib_opf_case3_lmbd.m"), curve = [1.0, 0.95]
            args = (path, curve, 2, Float64, nothing, 0.1, nothing)
            actual = ExaModelsPower.mpopf_args(args...)
            signature = Tuple{
                String,
                Vector{Float64},
                Int,
                Type{Float64},
                Nothing,
                Float64,
                Nothing,
            }
            inferred = Base.infer_return_type(ExaModelsPower.mpopf_args, signature)
            @test isconcretetype(inferred)
            @test inferred === typeof(actual)
        end

        # A load curve scales the LOADS. A fixed bus shunt is fixed: case14 carries
        # bs = 19.0 at bus 9, and it is the same in every period.
        let net = ExaModelsPower.parse_mp_network(_case_path("pglib_opf_case14_ieee.m"))
            series = PowerIO.LoadSeries(net, [1.0, 0.5], Float64)
            d = ExaModelsPower.parse_mp_power_data(net, series, 2, 0.1, Float64)
            shunt = [b.b.bs for b in d.busarray[:, 1]]
            @test any(!iszero, shunt)
            @test shunt == [b.b.bs for b in d.busarray[:, 2]]
            @test [b.b.pd for b in d.busarray[:, 2]] ≈ 0.5 .* [b.b.pd for b in d.busarray[:, 1]]
        end

        # A zero line rating means "no limit" in every format PowerIO reads, and real
        # cases use it: every arc of stock MATPOWER case14, case57 and case118. Read
        # literally it bounds the flow at exactly zero and asks for p^2 + q^2 <= 0,
        # which made those cases INFEASIBLE_PROBLEM_DETECTED rather than solved.
        mktempdir() do dir
            src = read(_case_path("pglib_opf_case14_ieee.m"), String)
            zeroed = joinpath(dir, "case14_norate.m")
            open(zeroed, "w") do io
                inbranch = false
                for l in split(src, '\n')
                    if !inbranch && occursin(r"^\s*mpc\.branch\s*=\s*\[", l)
                        inbranch = true
                    elseif inbranch && occursin(r"^\s*\];", l)
                        inbranch = false
                    elseif inbranch
                        f = split(l, '\t')
                        length(f) > 6 && (f[7] = " 0.0")
                        l = join(f, '\t')
                    end
                    println(io, l)
                end
            end

            d = ExaModelsPower.opf_args(zeroed; T = Float64)[1]
            @test all(isinf, d.rate_a)
            @test all(isinf, d.branch_rate_a)
            @test all(isinf, d.branch_thermal_ucon)

            for form in (Polar(), Rect())
                m, _, _ = ac_opf_model(zeroed; form = form)
                @test all(isfinite, NLPModels.cons(m, m.meta.x0))
                r = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-8)
                @test r.status == MadNLP.SOLVE_SUCCEEDED
            end

            # The rated case is untouched: every bound is the rating itself.
            rated = ExaModelsPower.opf_args(_case_path("pglib_opf_case14_ieee.m"); T = Float64)[1]
            @test all(isfinite, rated.rate_a)
            @test all(iszero, rated.branch_thermal_ucon)
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

            @test !isnothing(ac_opf_model(psse_path; form = Polar())[1])
            @test !isnothing(ac_opf_model(powerworld_path; form = Rect())[1])
            @test !isnothing(dcopf_model(psse_path)[1])
            @test !isnothing(
                mpopf_model(
                    powermodels_path,
                    [1.0, 0.95];
                    from = "powermodels",
                )[1],
            )

            # Passing only one of pd/qd is a clear error, not a MethodError from a
            # `nothing` reaching PowerIO.LoadSeries. Both are read from file, or both
            # are supplied as matrices; the files here are never read (the error is
            # raised before load construction).
            @test_throws ArgumentError mpopf_model(
                matpower, "unused.Pd", "unused.Qd"; pd = zeros(3, 2))
            @test_throws ArgumentError mpopf_model(
                matpower, "unused.Pd", "unused.Qd"; qd = zeros(3, 2))
        end
    end
end
