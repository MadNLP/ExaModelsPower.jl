import NLPModels

# The recipe, the eagerly-built core, and the model a caller actually gets must
# all be the same model. They are three spellings of one body, and the point of
# the split is that they cannot drift — so that is checked here rather than
# assumed.

function test_recipe_equivalence(filename, form; T = Float64)
    # recipe ∘ args — which is what `ac_opf_model` is
    m1, vars, _ = ac_opf_model(filename; form = form, T = T)
    # the same body built eagerly, data in hand — what `ExaModelsC` compiles
    core, _, _ = ac_opf_core(filename; form = form, T = T)
    m2 = ExaModel(core)

    @test m1.meta.nvar == m2.meta.nvar
    @test m1.meta.ncon == m2.meta.ncon
    @test m1.meta.nnzj == m2.meta.nnzj
    @test m1.meta.nnzh == m2.meta.nnzh
    @test Array(m1.meta.x0) == Array(m2.meta.x0)
    @test Array(m1.meta.lvar) == Array(m2.meta.lvar)
    @test Array(m1.meta.uvar) == Array(m2.meta.uvar)
    @test Array(m1.meta.lcon) == Array(m2.meta.lcon)
    @test Array(m1.meta.ucon) == Array(m2.meta.ucon)

    # Evaluated away from x0, so a difference in the starting point and one in
    # the algebra cannot hide behind one another.
    x = T[1 + T(0.01) * sin(i) for i = 1:m1.meta.nvar]
    y = T[T(0.01) * cos(i) for i = 1:m1.meta.ncon]
    @test NLPModels.obj(m1, x) == NLPModels.obj(m2, x)
    @test NLPModels.grad(m1, x) == NLPModels.grad(m2, x)
    @test NLPModels.cons(m1, x) == NLPModels.cons(m2, x)
    @test NLPModels.jac_coord(m1, x) == NLPModels.jac_coord(m2, x)
    @test NLPModels.hess_coord(m1, x, y; obj_weight = T(0.7)) ==
          NLPModels.hess_coord(m2, x, y; obj_weight = T(0.7))
end

# REGRESSION. The handles `ac_opf_model` returns come out of the RECIPE, where
# an offset is still an `ArgNode` expression rather than a number. Unless they
# are resolved against the arguments the model was built from, `solution`
# indexes the solution vector with a symbolic range and throws
#
#     ArgumentError: invalid index: (((0 + (_length(length(arg.bus)) * 1)) + …
#
# a long way from the cause. This shipped and the suite caught it.
function test_solution_handles(filename, form)
    m, vars, _ = ac_opf_model(filename; form = form)
    result = madnlp(m; print_level = MadNLP.ERROR, tol = 1e-6)
    @test result.status == MadNLP.SOLVE_SUCCEEDED
    # Each call throws outright on an un-instantiated handle; the lengths then
    # say the blocks are the ones they claim to be, and together they must
    # account for every variable in the model.
    total = 0
    for k in keys(vars)
        s = Array(solution(result, vars[k]))
        @test !isempty(s)
        total += length(s)
    end
    @test total == m.meta.nvar
end

# Compiling costs ~90 s per form and needs ExaModelsC, which is not a test
# dependency, so it is opt-in: EMP_TEST_AOT=1.
function test_aot(filename, form)
    @eval using ExaModelsC
    dir = mktempdir()
    core, _, _ = ac_opf_recipe(; form = form)
    r = Base.invokelatest(
        getfield(Main, :ExaModelsC).compile_library,
        joinpath(dir, "acopf"), core, filename; prefix = "acopf", argfun = ac_opf_args)
    @test isfile(r.libpath)
    rm(dir; recursive = true, force = true)
end
