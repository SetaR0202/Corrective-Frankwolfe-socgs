using FrankWolfe
using JSON
using Random
using StableRNGs
using HiGHS

include("tensor_completion.jl")

const side_length = 10
const tensor_order = 3
const use_standard_completion = true

for radius in (3.0,)
    for tensor_rank in (5, 10, 50)
        for seed in (1, 2)

            rng = StableRNG(seed)
            tensor_truth = compute_ground_truth(radius, tensor_rank, tensor_order, side_length; rng=rng)
            selected_indices = unique(rand(rng, eachindex(tensor_truth), 100))

            varweights = if use_standard_completion
                ones(length(selected_indices))
            else
                10 * exp.(randn(length(selected_indices)))
            end
            f, grad! = build_completion_function_gradient(tensor_truth, selected_indices, varweights)
            H = build_tensor_completion_hessian(tensor_truth, selected_indices, varweights)

            lmo = TensorLMO{tensor_order}(radius, side_length)
            x0 = FrankWolfe.compute_extreme_point(lmo, ones(size(tensor_truth)))

            FrankWolfe.blended_conditional_gradient(f, grad!, lmo, vec(x0), trajectory=true, max_iteration=1, line_search=FrankWolfe.Secant())
            res_bcg = FrankWolfe.blended_conditional_gradient(f, grad!, lmo, vec(x0), trajectory=true, epsilon=1e-5, line_search=FrankWolfe.Secant())

            FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, vec(x0), trajectory=true, max_iteration=1, lazy=true, line_search=FrankWolfe.Secant())
            res_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, vec(x0), trajectory=true, lazy=true, epsilon=1e-5, line_search=FrankWolfe.Secant())

            # b_term to use for the quadratic active set
            b_term = zero(vec(tensor_truth))
            b_term[selected_indices] .= - varweights .* tensor_truth[selected_indices] / length(selected_indices)

            FrankWolfe.blended_pairwise_conditional_gradient(
                f, grad!, lmo,
                FrankWolfe.ActiveSetQuadraticLinearSolve(
                    FrankWolfe.ActiveSet([(1.0, copy(vec(x0)))]),
                    H, b_term,
                    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                    wolfe_step=true,
                ),
                trajectory=true,
                max_iteration=1,
                lazy=true,
                line_search=FrankWolfe.Secant(),
            )
            res_direct_solve_wolfe = FrankWolfe.blended_pairwise_conditional_gradient(
                f, grad!, lmo,
                FrankWolfe.ActiveSetQuadraticLinearSolve(
                    FrankWolfe.ActiveSet([(1.0, copy(vec(x0)))]),
                    H, b_term,
                    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                    wolfe_step=true,
                ),
                trajectory=true,
                lazy=true,
                epsilon=1e-5,
                line_search=FrankWolfe.Secant(),
            )

            FrankWolfe.blended_pairwise_conditional_gradient(
                f, grad!, lmo,
                FrankWolfe.ActiveSetQuadraticLinearSolve(
                    FrankWolfe.ActiveSet([(1.0, copy(vec(x0)))]),
                    H, b_term,
                    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                    wolfe_step=false,
                ),
                trajectory=true,
                max_iteration=1,
                lazy=true,
                line_search=FrankWolfe.Secant(),
            )
            res_direct_solve = FrankWolfe.blended_pairwise_conditional_gradient(
                f, grad!, lmo,
                FrankWolfe.ActiveSetQuadraticLinearSolve(
                    FrankWolfe.ActiveSet([(1.0, copy(vec(x0)))]),
                    H, b_term,
                    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                    wolfe_step=false,
                ),
                trajectory=true,
                lazy=true,
                epsilon=1e-5,
                line_search=FrankWolfe.Secant(),
            )
            weighted_suffix = use_standard_completion ? "" : "_weighted"
            open("tensor_experiment_lazy_rank_$(tensor_rank)_radius_$(radius)_size_$(side_length)_$(seed)$(weighted_suffix).json", "w") do file
                write(
                    file,
                    JSON.json(
                        (
                            traj_bpcg = res_bpcg.traj_data, traj_bcg = res_bcg.traj_data,
                            traj_ds = res_direct_solve.traj_data, traj_ds_wolfe = res_direct_solve_wolfe.traj_data,
                            side_length = side_length, tensor_rank = tensor_rank, radius = radius,
                        )
                    ),
                )
            end
        end
    end
end
