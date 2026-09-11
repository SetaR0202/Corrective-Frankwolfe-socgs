using FrankWolfe
import MathOptInterface as MOI
using HiGHS

import FrankWolfe: ActiveSet
import FrankWolfe: LinearMinimizationOracle
import FrankWolfe: CorrectiveStep
import FrankWolfe: LineSearchMethod
import FrankWolfe: get_active_set_iterate
import FrankWolfe: corrective_frank_wolfe
import FrankWolfe: fast_dot
import FrankWolfe: AbstractActiveSet
import FrankWolfe: Adaptive
import FrankWolfe: Secant

"""
`Base.copyto!(dest::AbstractActiveSet{AT,R,IT},src::AbstractActiveSet{AT,R,IT})`
Overwrite the weight and atom vectors and the iterate of `dest`
by those of `src`. 
If necessary, resizing of the vectors in `dest.weights` and `dest.atoms` are performed.
However, if `dest.x` and `src.x` do not have the same size, then a `DimensionMismatch` is thrown.
"""
function Base.copyto!(dest::AbstractActiveSet{AT,R,IT},src::AbstractActiveSet{AT,R,IT}) where {AT,R,IT}
    if size(dest.x) != size(src.x)
        throw(DimensionMismatch("Different dimensions for x."))
    end
    if size(dest) != size(src)
        s = size(src)[1]
        resize!(dest.weights,s)
        resize!(dest.atoms,s)
    end
    copyto!(dest.weights,src.weights)
    copyto!(dest.atoms,src.atoms) 
    copyto!(dest.x,src.x)
    return dest
end

"""
    ActiveSetQuadraticLinearSolve(tuple_values::Vector{Tuple{R,AT}}, A, b, lp_optimizer)

Creates an `ActiveSetQuadraticLinearSolve` from the given Hessian `A`, linear term `b` and `lp_optimizer` by creating an inner `ActiveSetQuadraticProductCaching` active set.
and counter parameter
"""
function FrankWolfe.ActiveSetQuadraticLinearSolve(
    tuple_values::Vector{Tuple{R,AT}},
    A::H,
    b,
    lp_optimizer;
    scheduler=LogScheduler(),
    wolfe_step=false,
    counter::Base.RefValue{Int} = Ref(0)
) where {AT,R,H}
    inner_as = FrankWolfe.ActiveSetQuadraticProductCaching(tuple_values, A, b)
    return FrankWolfe.ActiveSetQuadraticLinearSolve(
        inner_as.weights,
        inner_as.atoms,
        inner_as.x,
        inner_as.A,
        inner_as.b,
        inner_as,
        lp_optimizer,
        wolfe_step,
        scheduler,
        counter,
    )
end

#################################################################################################################
#################################################################################################################
#SOCGS  Second-order conditional gradient sliding, Carderera, Alejandro and Pokutta, Sebastian, arXiv preprint arXiv:2002.08907
#################################################################################################################
#################################################################################################################

@enum CGSStepsize begin
    CGS_FW_STEP = 1
    CGS_PVM_STEP = 2
end


abstract type LowerBoundEstimator end
function compute_pvm_threshold end

struct LowerBoundFiniteSteps{LS<:LineSearchMethod,R<:Real} <: LowerBoundEstimator 
    corrective_step::CorrectiveStep
    max_iter::Int
    line_search::LS
    min_threshold::R
end

LowerBoundFiniteSteps(f,grad!,
                    lmo,
                    corrective_step,
                    max_iter,
                    line_search
                    ) = LowerBoundFiniteSteps(f,grad!,lmo, corrective_step, max_iter,line_search, 1e-4)
            
function compute_pvm_threshold(lb_estimator::LowerBoundFiniteSteps,f,grad!,lmo,
                                x,
                                primal::Real,
                                gradient,
                                dual_gap
                                )
    _, _, primal_finite_steps, _, _, _ =  corrective_frank_wolfe(
                            f,
                            grad!,
                            lmo,
                            lb_estimator.corrective_step,
                            ActiveSet([(one(x[1]),x)]);
                            line_search = lb_estimator.line_search,
                            max_iteration= lb_estimator.max_iter -1 , #-1 because recompute_last_vertex = true
    )
    return max((primal - primal_finite_steps)^4/(fast_dot(gradient,gradient)^2),lb_estimator.min_threshold)
end


struct LowerBoundLSmoothness{LS<:LineSearchMethod,R<:Real} <: LowerBoundEstimator 
    corrective_step::CorrectiveStep
    max_iter::Int
    line_search::LS
    min_threshold::R
    L::R
end


function compute_pvm_threshold(lb_estimator::LowerBoundLSmoothness,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )

    function make_linear_search_stepsize_callback(traj_data::Vector)
        return function callback_with_trajectory(state, args...)
            if state.step_type !== FrankWolfe.ST_LAST || state.step_type !== FrankWolfe.ST_POSTPROCESS
                push!(traj_data, state.gamma)
            end
            return true
        end
    end
        gamma_traj = []    
        _, v, primal_finite_steps, dual_gap, _, _ =  corrective_frank_wolfe(
        f,
        grad!,
        lmo,
        lb_estimator.corrective_step,
        ActiveSet([(one(x[1]),x)]);
        line_search = lb_estimator.line_search,
        max_iteration= lb_estimator.max_iter -1 , #-1 because recompute_last_vertex = true
        callback = make_linear_search_stepsize_callback(gamma_traj)
    )
    L = lb_estimator.L
    gamma = gamma_traj[end]
    norm2_v_x = FrankWolfe.fast_dot(v,v) - 2.0 * FrankWolfe.fast_dot(v,x) + FrankWolfe.fast_dot(x,x)
    return max(gamma* dual_gap - 0.5*L * gamma^2 * norm2_v_x,lb_estimator.min_threshold)
end


struct LowerBoundKnown{R<:Real} <: LowerBoundEstimator 
    known_optimal_sol::R
    min_threshold::R
end
LowerBoundKnown(known_optimal_sol) = LowerBoundKnown(known_optimal_sol, 1e-7)

function compute_pvm_threshold(lb_estimator::LowerBoundKnown,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    return max( (primal - lb_estimator.known_optimal_sol)^4/(fast_dot(gradient,gradient)^2), lb_estimator.min_threshold)
end

struct LowerBoundConstant{R<:Real} <: LowerBoundEstimator 
    lowerbound_value::R
end 


function compute_pvm_threshold(lb_estimator::LowerBoundConstant,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    return lb_estimator.lowerbound_value
end


struct LowerBoundDualGapSquared{R<:Real} <: LowerBoundEstimator 
    factor::R
    default_threshold::R
end 


function compute_pvm_threshold(lb_estimator::LowerBoundDualGapSquared,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    if dual_gap < Inf
        return lb_estimator.factor* dual_gap^2
    else
        return lb_estimator.default_threshold
    end
end


struct LowerBoundDualGapLinear{R<:Real} <: LowerBoundEstimator 
    factor::R
    default_threshold::R
end 


function compute_pvm_threshold(lb_estimator::LowerBoundDualGapLinear,f,grad!,lmo,
    x,
    primal::Real,
    gradient,
    dual_gap
    )
    if dual_gap < Inf
        return lb_estimator.factor * dual_gap
    else
        return lb_estimator.default_threshold
    end
end

###########################################################################

#=
function current_quad_approx(primal,gradient,x,
                    Hx,
                    gradient_corrector!,
                    gradient_corrector_storage,
                    quadratic_term_function!,quadratic_term_storage
                    )
    constant_term = primal - FrankWolfe.fast_dot(gradient,x) + 0.5 * FrankWolfe.fast_dot(Hx,x)
    function f_quad_approx(p)
        res = 0.5*quadratic_term_function!(quadratic_term_storage,p) + FrankWolfe.fast_dot(gradient,p) - FrankWolfe.fast_dot(Hx,p) + constant_term 
      return res
    end
    function grad_quad_approx!(storage,p)
        gradient_corrector!(gradient_corrector_storage,p)
        storage .= gradient + gradient_corrector_storage
    end

    return f_quad_approx,grad_quad_approx!
end
=#

function make_pvm_callback(callback, traj_data::Vector)
    return function callback_with_trajectory(state, args...)
        if state.step_type !== FrankWolfe.ST_LAST || state.step_type !==  FrankWolfe.ST_POSTPROCESS
            push!(traj_data, (pvm_t = state.t, pvm_time = state.time ) )
        end
        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end


function second_order_conditional_gradient_sliding(
    f,
    grad!,
    build_quadratic_approximation!, 
    fw_step::CorrectiveStep, 
    lmo_fw::LinearMinimizationOracle,
    pvm_step::CorrectiveStep, 
    lmo_pvm::LinearMinimizationOracle,
    x0;
    lb_estimator::LowerBoundEstimator,
    line_search_fw::LineSearchMethod=Secant(),
    line_search_pvm::LineSearchMethod=Secant(),
    line_search_after_pvm::Union{Nothing,LineSearchMethod}=Secant(),
    memory_mode_after_pvm::FrankWolfe.MemoryEmphasis=FrankWolfe.InplaceEmphasis(),
    line_search_workspace_after_pvm=nothing,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    verbose_pvm = false,
    traj_data=[],
    pvm_traj_data=[],
    timeout=Inf,
    pvm_max_iteration,
    pvm_with_quadratic_active_set = false,
    H_quadratic_active_set = nothing,
    b_quadratic_active_set = nothing,
    scaling_factor = 10,
    #
    pvm_trajectory = false, #DEBUG
    lazy_pvm =false,
    do_wolfe = true,
    #
    do_cgs= false
)


    active_set_fw = ActiveSet([(one(x0[1]),x0)])
    active_set_pvm = ActiveSet([(one(x0[1]),x0)]) 
    gradient= collect(x0)
    Hx = collect(x0)
    x_fw = get_active_set_iterate(active_set_fw)
    x_pvm = get_active_set_iterate(active_set_pvm)
    if line_search_after_pvm !== nothing
        x_pvm_copy = collect(x0)
        direction_linesearch_after_pvm = collect(x0)
        gradient_pvm = similar(x0)
    end
    x = x_pvm    
    dual_gap_fw = Inf
    socgs_fw_gap = Inf
    primal_fw = Inf
    primal_pvm = Inf
    primal = f(x0)

    t = 0
    step_type = CGS_FW_STEP
    time_start = time_ns()
    tot_time = 0.0
    lmo_pvm = FrankWolfe.TrackingLMO(lmo_pvm) #wrapper to track lmos call in pvm

    if trajectory
        state = (
                t= t,
                primal = primal,
                dual_fw = Inf,
                dual_gap_fw = Inf,
                tot_time = tot_time,          
                x = x,
                epsilon = Inf,
                step_type = step_type,
                grad_time = 0.0,
                quad_time = 0.0,
                thresh_time = 0.0,
                pvm_t = 0,
                pvm_tot_time = 0.0,  
                primal_eval_time = 0.0,   
                fw_time = 0.0,         
                copyto_time = 0.0,
                traj_data_pvm = [],
                cumul_lmo_calls_pvm = 0
            )
        push!(
            traj_data,
            state
        )
    end


    function relative_gap_stop_condition(::Any,primal_value)
        return true
    end
    #=
    function relative_gap_stop_condition(lb_estimator::LowerBoundKnown,primal_value)
        rel_gap = abs(primal_value - lb_estimator.known_optimal_sol )/ (1e-8 + abs(lb_estimator.known_optimal_sol))
    return rel_gap > 1e-12
    end
    =#
    #xold = copy(x0)#DEBUG
    #xold_fw = copy(x0)#DEBUG
    while t < max_iteration && tot_time ≤ timeout && relative_gap_stop_condition(lb_estimator,primal)
        #pvm_traj_data
        pvm_traj_data = []
        pvm_callback = make_pvm_callback(nothing, pvm_traj_data)

        #computing gradient 
        grad_time_start = time_ns()
        grad!(gradient, x)
        grad_time = (time_ns() - grad_time_start ) / 1e9

        #dual gap of fw
        v = FrankWolfe.compute_extreme_point(lmo, gradient)
        socgs_fw_gap = dot(gradient,x-v)
        #building quadratic approximation (problem dependent) 
        build_quad_time_start = time_ns()

        #quadratic_term_function!, gradient_corrector!, Hx = build_quadratic_approximation!(Hx,x,gradient,primal)
        f_quad_approx, grad_quad_approx! = build_quadratic_approximation!(Hx,x,gradient,primal, H_quadratic_active_set, b_quadratic_active_set,t+1)
        build_quad_tot_time = (time_ns() - build_quad_time_start) / 1e9
              
        #compute threshold for pvm
        threshold_time_start = time_ns()
        #@info "DEBUG===================================" t state.dual_gap_fw
        epsilon = compute_pvm_threshold(lb_estimator,f,grad!,lmo_fw,x,primal,gradient,state.dual_gap_fw) 
        threshold_tot_time = (time_ns() - threshold_time_start) / 1e9
        #H-projection (pvm)
        if line_search_after_pvm !== nothing
            x_pvm_copy .= x
        end
        if pvm_with_quadratic_active_set
            
            #building approximation
            active_set_quadratic_wolfe = FrankWolfe.ActiveSetQuadraticLinearSolve(
                active_set_pvm, #ActiveSetQuadraticProductCaching(active_set_pvm.weights, active_set_pvm.atoms #USE tuples (weight, atom), H_quadratic_active_set,gradient)
                H_quadratic_active_set, b_quadratic_active_set,
                MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                scheduler=FrankWolfe.LogScheduler(start_time = t< 25 ? pvm_max_iteration : 1  , scaling_factor=scaling_factor),
                wolfe_step=do_wolfe,
            )            
            #direct solve pvm
            if isa(pvm_step,FrankWolfe.BlendedPairwiseStep)
                x_pvm, _, _,_, traj_data_pvm, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f_quad_approx,
                    grad_quad_approx!,
                    lmo_pvm,
                    active_set_quadratic_wolfe,
                    line_search=line_search_pvm,
                    epsilon = epsilon,
                    callback=pvm_callback,
                    trajectory= pvm_trajectory,
                    lazy = lazy_pvm,
                    verbose = verbose_pvm, 
                    max_iteration = pvm_max_iteration,
                    print_iter = 100,
                    sparsity_control=pvm_step.lazy_tolerance,
                );
            elseif isa(pvm_step,FrankWolfe.AwayStep)
                x_pvm, _, _,_, traj_data_pvm, _ = FrankWolfe.away_frank_wolfe(
                    f_quad_approx,
                    grad_quad_approx!,
                    lmo_pvm,
                    active_set_quadratic_wolfe,
                    line_search=line_search_pvm,
                    epsilon = epsilon,
                    callback=pvm_callback,
                    trajectory= pvm_trajectory,
                    lazy = lazy_pvm,
                    verbose = verbose_pvm, 
                    max_iteration = pvm_max_iteration,
                    print_iter = 100,
                    lazy_tolerance= pvm_step.lazy_tolerance,
                );
            else
                throw(ErrorException("Direct solve not implemented for "*string(typeof(pvm_step ))*"."))
            end
        else
            #pvm
            x_pvm, _ , _,_, traj_data_pvm, _ = corrective_frank_wolfe(
                    f_quad_approx,
                    grad_quad_approx!,
                    lmo_pvm,
                    pvm_step,
                    active_set_pvm;
                    line_search=line_search_pvm,
                    epsilon= epsilon, 
                    callback = pvm_callback,
                    trajectory= pvm_trajectory,
                    verbose = verbose_pvm, 
                    max_iteration = pvm_max_iteration ,
                )
        end
        
        #Performing linesearch on pvm solution
        if line_search_workspace_after_pvm ===nothing && line_search_after_pvm !== nothing  
            line_search_workspace_after_pvm = FrankWolfe.build_linesearch_workspace(line_search_after_pvm,x_pvm,gradient_pvm)   
        end
        if line_search_after_pvm !== nothing
            direction_linesearch_after_pvm .= x_pvm .- x_pvm_copy
            #grad!(gradient_pvm,x_pvm)
            gamma = FrankWolfe.perform_line_search(
                line_search_after_pvm ,
                t+1,
                f, 
                grad!,
                gradient,
                x_pvm,
                direction_linesearch_after_pvm,
                one(eltype(x_pvm)), #max gamma
                line_search_workspace_after_pvm,
                memory_mode_after_pvm,
            )
            x_pvm .= x_pvm_copy .+ gamma *(direction_linesearch_after_pvm) 
        end

        #primal evaluation
        primal_eval_time_start = time_ns()
        primal_pvm = f(x_pvm)
        primal_eval_time = (time_ns() - primal_eval_time_start) / 1e9

        #Fw corrective step
        fw_step_time_start = time_ns()
        x_fw, _ , primal_fw, dual_gap_fw , _=corrective_frank_wolfe(
                f,
                grad!,
                lmo_fw,
                fw_step,
                active_set_fw;
                line_search=line_search_fw,
                max_iteration=0, #0 because recompute_last_vertex = true
                gradient = gradient
            )
        fw_step_tot_time = (time_ns() - fw_step_time_start) / 1e9

        #copying active sets
        atoms_pvm = sum(x_pvm .!= 0)
        atoms_fw = sum(x_fw .!= 0)
        #@info "DEBUG" length(traj_data_pvm) primal_pvm primal_fw atoms_pvm atoms_fw
        #xold .= x_pvm#DEBUG
        #xold_fw .= x_fw#DEBUG
        copyto_time_start = time_ns()
        @info "DEBUG" primal_pvm primal_fw
        if (!do_cgs) && primal_pvm > primal_fw
            copyto!(active_set_pvm,active_set_fw)
            primal = primal_fw
            x = x_fw
            step_type = CGS_FW_STEP
        else
            #copyto!(active_set_fw,active_set_pvm)
            primal = primal_pvm
            x = x_pvm
            step_type = CGS_PVM_STEP
        end
        copyto_tot_time = (time_ns() - copyto_time_start) / 1e9

        tot_time = (time_ns() - time_start) / 1e9
        state = (
                t= t,
                primal = primal,
                dual_fw = primal_fw - socgs_fw_gap,
                dual_gap_fw = socgs_fw_gap,
                tot_time = tot_time,          
                x = x,
                epsilon = epsilon,
                step_type = step_type,
                grad_time = grad_time,
                quad_time = build_quad_tot_time,
                thresh_time = threshold_tot_time,
                pvm_t = pvm_traj_data[end].pvm_t,
                pvm_tot_time = pvm_traj_data[end].pvm_time,  
                primal_eval_time = primal_eval_time,   
                fw_time = fw_step_tot_time,         
                copyto_time = copyto_tot_time,
                traj_data_pvm = traj_data_pvm,
                cumul_lmo_calls_pvm = lmo_pvm.counter
            )
        t += 1
        if trajectory
            push!(
                traj_data,
                state
            )
        end

        if verbose && mod(t, print_iter) == 0
            rel_gap_string =""
            if isa(lb_estimator,LowerBoundKnown)
                rel_gap = abs(primal - lb_estimator.known_optimal_sol )/ (1e-8 + abs(lb_estimator.known_optimal_sol))
                rel_gap_string = " Rel.Gap "*string(rel_gap)
            end

            println("It. ", t, " Tot.Time ", tot_time," ", step_type,rel_gap_string)
        end

        
    end

    if verbose && timeout < Inf && tot_time ≥ timeout
        @info "Time limit reached"
    end

    if verbose && t ≥ max_iteration
        @info "Iteration limit reached"
    end


    return (x=x, primal=primal, dual_gap_fw = socgs_fw_gap,
             t = t, tot_time = tot_time,traj_data=traj_data)
end