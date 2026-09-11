function build_trajectory_callback(callback, traj_data::Vector)
    return function callback_with_trajectory(state, args...)
        if state.step_type !== FrankWolfe.ST_LAST && state.step_type !== FrankWolfe.ST_POSTPROCESS
            push!(traj_data, (state.t, state.primal, state.dual, state.dual_gap, state.time, length(args[1])))
        end
        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end

function bpcg_run(f, grad!, lmo, as; callback=nothing, kwargs...)

    traj_data = []
    callback = build_trajectory_callback(callback, traj_data)

    FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        as;
        lazy=true,
        callback=callback,
        line_search=FrankWolfe.Secant(),
        kwargs...
    )

    return traj_data
end

function build_quad_as(x0, quad_term, linear_term)
    return FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], quad_term, linear_term)
end

function build_quad_as_linear_solve(x0, quad_term, linear_term, start_time, scaling_factor, wolfe)
    return FrankWolfe.ActiveSetQuadraticLinearSolve(
        build_quad_as(x0, quad_term, linear_term),
        quad_term, linear_term, MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
        scheduler=FrankWolfe.LogScheduler(start_time=start_time, scaling_factor=scaling_factor),
        wolfe_step=wolfe,
    )
end

function run_qc_comparison(build_problem, params, start_time, scaling_factor; kwargs...)

    f, grad!, lmo, x0, quad_term, linear_term = build_problem(params...)

    t_BPCG = bpcg_run(f, grad!, lmo, build_quad_as(x0, quad_term, linear_term); kwargs...)
    t_DSW = bpcg_run(f, grad!, lmo, build_quad_as_linear_solve(x0, quad_term, linear_term, start_time, scaling_factor, true); kwargs...)
    t_DSLP = bpcg_run(f, grad!, lmo, build_quad_as_linear_solve(x0, quad_term, linear_term, start_time, scaling_factor, false); kwargs...)

    return t_BPCG, t_DSW, t_DSLP
end
