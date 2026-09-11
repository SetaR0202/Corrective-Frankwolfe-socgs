

struct ShiftedLMO{T} <: FrankWolfe.LinearMinimizationOracle
    lmo::FrankWolfe.LinearMinimizationOracle
    center::T
end

function FrankWolfe.compute_extreme_point(lmo::ShiftedLMO{T}, direction::T) where {T}
    return FrankWolfe.compute_extreme_point(lmo.lmo, direction) .+ lmo.center
end


"""
    Define the line search methods
"""

struct NewSplitStepSize <: FrankWolfe.LineSearchMethod end

function FrankWolfe.perform_line_search(
    ::NewSplitStepSize,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
)
    return 2.0 / (sqrt(t + 2) * log(t + 2))
end

struct OldSplitStepSize <: FrankWolfe.LineSearchMethod end

function FrankWolfe.perform_line_search(
    ::OldSplitStepSize,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
)
    return 2.0 / (sqrt(t) + 2.0)
end


"""
    Define the lambda schedules
"""

lambda_func_new(λ0=1.0) = (state) -> 1.0 / log(state.t + 2) * log(2) / λ0
lambda_func_old(λ0=1.0) = (state) -> 1.0 / (1.0 / state.f.λ[] + λ0 / (sqrt(state.t) + 2)^2)



"""
    Extended the log scheduler with minimum size of active set
"""
struct LogSchedulerMinSize{T}
    start_time::Int
    scaling_factor::T
    max_interval::Int
    current_interval::Base.RefValue{Int}
    last_solve_counter::Base.RefValue{Int}
    min_size::Int
end

LogSchedulerMinSize(; start_time=20, scaling_factor=1.5, max_interval=1000, min_size=0) =
    LogSchedulerMinSize(start_time, scaling_factor, max_interval, Ref(start_time), Ref(0), min_size)

function FrankWolfe.should_solve_lp(as::FrankWolfe.ActiveSetQuadraticLinearSolve, scheduler::LogSchedulerMinSize)
    if length(as.active_set) >= scheduler.min_size && as.counter[] - scheduler.last_solve_counter[] >= scheduler.current_interval[]
        scheduler.last_solve_counter[] = as.counter[]
        scheduler.current_interval[] = min(
            round(Int, scheduler.scaling_factor * scheduler.current_interval[]),
            scheduler.max_interval,
        )
        return true
    end
    return false
end


"""
    Define callback functions
"""

# Callback for printing the active set sizes
function build_print_callback(step_1, callback, print_iter)

    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Dist2", "#AS 1"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14e %13s\n"

    function format_state(state, args...)
        rep = (
            FrankWolfe.steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            Float64(0.5 * sum(abs2, state.x.blocks[1] - state.x.blocks[2])),
            string(length(step_1.active_set)),
        )
        return rep
    end

    return FrankWolfe.make_print_callback(callback, print_iter, headers, format_string, format_state)
end


# Function for building direct solve steps
function build_qc_step(as, A, b, wolfe; start_time=100, scaling_factor=1.5, max_interval=1000, sparsity_control=2.0, min_size=0)
    return FrankWolfe.BPCGStep(true, FrankWolfe.ActiveSetQuadraticLinearSolve(
            as,
            A,
            b,
            MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
            scheduler=LogSchedulerMinSize(start_time=start_time, scaling_factor=scaling_factor, max_interval=max_interval, min_size=min_size),
            wolfe_step=wolfe,
        ), 1000, sparsity_control, Inf)
end


# Function for building callback that updates the quadratic model
function make_update_as_callback(step_1, quad_matrix, quad_factor, linear_term, start_time, scaling_factor, max_interval, callback, wolfe, min_size)
    return function (state, args...)
        y = copy(state.x.blocks[2])
        l = state.f.λ[]
        c1 = step_1.active_set.counter[]
        c2 = step_1.active_set.scheduler.last_solve_counter[]

        # Update active scalar factor of active set
        as = step_1.active_set.active_set
        #as.λ[] = quad_factor(l)

        step_1.active_set = FrankWolfe.ActiveSetQuadraticLinearSolve(
            as,
            quad_matrix*quad_factor(l),
            linear_term(l, y),
            MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
            scheduler=LogSchedulerMinSize(
                start_time,
                scaling_factor,
                max_interval,
                Ref(start_time),
                Ref(c2), # Copy the last solve counter from previous step
                min_size
            ),
            wolfe_step=wolfe
        )

        # Update counter of active set
        step_1.active_set.counter[] = c1

        if callback !== nothing
            return callback(state, args...)
        end
        return true
    end
end


# Function for building trajectory callback
# Store addtionally the active ste size, number of lmo calls and distance between x1 and x2 
function make_trajectory_callback(trajectory, step, callback)
    return function (state, args...)
        as_size = step isa FrankWolfe.FrankWolfeStep ? 1 : length(step.active_set)
        tuple = (state.t, state.primal, state.primal - state.dual_gap, state.dual_gap, state.time, as_size, state.lmo.lmos[1].counter, 0.5 * sum(abs2, state.x.blocks[1] - state.x.blocks[2]))
        push!(trajectory, tuple)
        if callback !== nothing
            return callback(state, args...)
        end
        return true
    end
end


function run_qc_comparison_alm(
    build_problem,
    params,
    lambda_func;
    start_time=100,
    scaling_factor=1.0,
    max_interval=1000,
    min_size=0,
    verbose=true,
    max_iteration=10000,
    print_iter=max_iteration / 10,
    sparsity_control=2.0,
    kwargs...)

    # Build problem
    f, grad!, lmo1, lmo2, x01, x02, linear_term, quad_matrix, quad_factor = build_problem(params...)

    # Define single run of ALM
    function alm_run(lambda_func; kwargs...)
        lmo1.counter = 0
        FrankWolfe.alternating_linear_minimization(
            FrankWolfe.block_coordinate_frank_wolfe,
            f,
            grad!,
            (lmo1, lmo2), (copy(x01), copy(x02));
            lambda=lambda_func,
            update_order=FrankWolfe.CyclicUpdate(),
            print_iter=print_iter,
            kwargs...,
        )
    end


    # BPCG run setup
    bpcg_step_1 = FrankWolfe.BPCGStep(true, nothing, 1000, sparsity_control, Inf)
    bpcg_step_2 = FrankWolfe.FrankWolfeStep()

    trajectory_bpcg = []
    bpcg_callback = make_trajectory_callback(trajectory_bpcg, bpcg_step_1, nothing)

    if verbose
        bpcg_callback = build_print_callback(bpcg_step_1, bpcg_callback, print_iter)
    end


    # QC-MNP run setup
    qc_mnp_step_1 = build_qc_step(
        FrankWolfe.ActiveSet([(1.0, copy(x01))]),#FrankWolfe.ActiveSetPartialCaching([(1.0, copy(x01))], quad_matrix, quad_factor(1.0)),
        quad_matrix*quad_factor(1.0),
        linear_term(1.0, x02),
        true;
        start_time=start_time,
        scaling_factor=scaling_factor,
        max_interval=max_interval,
        sparsity_control=sparsity_control,
        min_size=min_size
    )
    qc_mnp_step_2 = FrankWolfe.FrankWolfeStep()

    trajectory_qc_mnp = []
    qc_mnp_callback = make_update_as_callback(qc_mnp_step_1, quad_matrix, quad_factor, linear_term, start_time, scaling_factor, max_interval, nothing, true, min_size)
    qc_mnp_callback = make_trajectory_callback(trajectory_qc_mnp, qc_mnp_step_1, qc_mnp_callback)

    if verbose
        qc_mnp_callback = build_print_callback(qc_mnp_step_1, qc_mnp_callback, print_iter)
    end


    # QC-LP run setup
    qc_lp_step_1 = build_qc_step(
        FrankWolfe.ActiveSet([(1.0, copy(x01))]),#FrankWolfe.ActiveSetPartialCaching([(1.0, copy(x01))], quad_matrix, quad_factor(1.0)),
        quad_matrix*quad_factor(1.0),
        linear_term(1.0, x02),
        false;
        start_time=start_time,
        scaling_factor=scaling_factor,
        max_interval=max_interval,
        sparsity_control=sparsity_control,
        min_size=min_size
    )
    qc_lp_step_2 = FrankWolfe.FrankWolfeStep()

    trajectory_qc_lp = []
    qc_lp_callback = make_update_as_callback(qc_lp_step_1, quad_matrix, quad_factor, linear_term, start_time, scaling_factor, max_interval, nothing, false, min_size)
    qc_lp_callback = make_trajectory_callback(trajectory_qc_lp, qc_lp_step_1, qc_lp_callback)

    if verbose
        qc_lp_callback = build_print_callback(qc_lp_step_1, qc_lp_callback, print_iter)
    end

    # Actual runs
    alm_run(lambda_func; update_step=(bpcg_step_1, bpcg_step_2), callback=bpcg_callback, max_iteration=max_iteration, kwargs...)
    alm_run(lambda_func; update_step=(qc_mnp_step_1, qc_mnp_step_2), callback=qc_mnp_callback, max_iteration=max_iteration, kwargs...)
    alm_run(lambda_func; update_step=(qc_lp_step_1, qc_lp_step_2), callback=qc_lp_callback, max_iteration=max_iteration, kwargs...)

    return trajectory_bpcg, trajectory_qc_mnp, trajectory_qc_lp
end