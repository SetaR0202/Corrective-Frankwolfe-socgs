using FrankWolfe
import LinearAlgebra as LA
using HiGHS
import MathOptInterface as MOI
using Plots

include("../src/entanglement_detection.jl")
include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))


# Define additional (full) scheduler besides default log scheduler
struct FullScheduler end

function FrankWolfe.should_solve_lp(::FrankWolfe.ActiveSetQuadraticLinearSolve, ::FullScheduler)
    return true
end


function entanglement_detection_experiment(a, v)

    T = Float64
    dims = (3, 3)

    # Horodecki state 3x3
    ρ = state_horodecki33(Complex{T}, a; v)
    C = correlation_tensor(ρ, dims)
    lmo = AlternatingSeparableLMO(T, dims; nb = 10, threshold = Base.rtoldefault(T), max_iter = 10^3)


    ini_sigma = Matrix{Complex{T}}(LA.I, prod(lmo.dims), prod(lmo.dims)) / prod(lmo.dims)
    ini_tensor = correlation_tensor(ini_sigma, lmo.dims)
    x0 = FrankWolfe.compute_extreme_point(lmo, ini_tensor - C)

    # Define a single run of separable distance with fixed parameters
    single_run(active_set, lmo; max_iteration = 10^6) = separable_distance(C, lmo;
        fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
        verbose = 2,
        max_iteration = max_iteration,
        callback_iter = 10^6,
        recompute_last_vertex = false,
        epsilon = 1e-7,
        active_set = active_set,
        trajectory = true,
        shortcut = false,
    )

    active_set_qc_mnp = FrankWolfe.ActiveSetQuadraticLinearSolve(
        FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), copy(x0))], LA.I, -C),
        LA.I,
        -C,
        MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
        wolfe_step=true,
        scheduler = FrankWolfe.LogScheduler(start_time=0),
    )

    active_set_qc_lp = FrankWolfe.ActiveSetQuadraticLinearSolve(
        FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), copy(x0))], LA.I, -C),
        LA.I,
        -C,
        MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
        wolfe_step=false,
        scheduler = FrankWolfe.LogScheduler(start_time=10),
    )

    # Dummy runs for avoiding precompilation
    single_run(copy(active_set_qc_mnp), lmo; max_iteration = 10)
    single_run(copy(active_set_qc_lp), lmo; max_iteration = 10)
    single_run(nothing, lmo; max_iteration = 10)

    # Actual runs
    res_qc_mnp = single_run(active_set_qc_mnp, lmo)
    res_qc_lp = single_run(active_set_qc_lp, lmo)
    res_bpcg = single_run(nothing, lmo)

    return res_bpcg, res_qc_mnp, res_qc_lp
end

# Experiment settings
a = 0.5
v = 0.97

results = entanglement_detection_experiment(a, v)

plot_trajectories([r.traj_data for r in results], ["BPCG" "QC-MNP" "QC-LP"], marker_shapes = [:circle, :star5, :diamond], reduce_size= true, filename = "entanglement_detection_experiment_$(a)_$(v).pdf")

