using MathOptInterface
using SCIP
using FrankWolfe
using JuMP
using LinearAlgebra
using Random

"""
Tensor LMO for nonnegative rank completion tensors.
The tensors are of dimensions side_length × side_length × side_length... N times.
For instance N = 2 would create a square matrix.
"""
struct TensorLMO{N} <: FrankWolfe.LinearMinimizationOracle
    radius::Float64
    side_length::Int
    o::SCIP.Optimizer
    tensor_var::Array{MOI.VariableIndex, N}
    binvar_vector::Vector{Array{MOI.VariableIndex, N}}
end

function TensorLMO{N}(radius::Real, side_length=10) where {N}
    o = SCIP.Optimizer()
    nvar_tensor = side_length^N
    tensor_indices = ntuple(i -> side_length, N)
    tensor_var = reshape(MOI.add_variables(o, nvar_tensor), tensor_indices)
    MOI.add_constraint.(o, tensor_var, MOI.GreaterThan(0.0))
    MOI.add_constraint.(o, tensor_var, MOI.LessThan(radius))
    binvar_vector = map(1:N) do k
        theta = reshape(MOI.add_variables(o, nvar_tensor), tensor_indices)
        MOI.add_constraint.(o, theta, MOI.ZeroOne())
        theta
    end
    sum_binvars = sum(binvar_vector; init = zeros(tensor_indices))
    MOI.add_constraint.(o, radius * sum_binvars - tensor_var, MOI.LessThan(-radius * (1 - N)))
    for k in 1:N
        MOI.add_constraint.(o, tensor_var - radius * binvar_vector[k], MOI.LessThan(0.0))
    end
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(o, MOI.Silent(), true)
    return TensorLMO{N}(radius, side_length, o, tensor_var, binvar_vector)
end

function FrankWolfe.compute_extreme_point(lmo::TensorLMO{N}, direction::AbstractArray{T,N}; kwargs...) where {T, N}
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), sum(lmo.tensor_var .* direction))
    MOI.optimize!(lmo.o)
    resstat = MOI.get(lmo.o, MOI.TerminationStatus())
    if resstat != MOI.OPTIMAL
        @warn "result $resstat not optimal. Resulting vector might be invalid."
    end
    return MOI.get.(lmo.o, MOI.VariablePrimal(), lmo.tensor_var)
end

# flattened vectorized form
function FrankWolfe.compute_extreme_point(lmo::TensorLMO{N}, direction::AbstractVector{T}; kwargs...) where {N,T}
    tensor_indices = ntuple(i -> lmo.side_length, N)
    tensor_direction = reshape(direction, tensor_indices)
    tensor_vertex = FrankWolfe.compute_extreme_point(lmo, tensor_direction)
    return vec(tensor_vertex)
end

function compute_ground_truth(radius, tensor_rank, N, side_length; rng=Random.GLOBAL_RNG)
    tensor_indices = ntuple(i -> side_length, N)
    lmo = TensorLMO{N}(radius, side_length)
    vertices = map(1:tensor_rank) do _
        d = randn(tensor_indices)
        FrankWolfe.compute_extreme_point(lmo, d)
    end
    conv_weights = rand(tensor_rank)
    conv_weights ./= sum(conv_weights)
    return sum(vertices[k] * conv_weights[k] for k in 1:tensor_rank)
end

function build_completion_function_gradient(tensor_truth, selected_indices, weights=ones(length(selected_indices)))
    function f(x)
        err = 0.0
        for (idx, tensor_idx) in enumerate(selected_indices)
            err += weights[idx] * (x[tensor_idx] - tensor_truth[tensor_idx])^2
        end
        return err / (2 * length(selected_indices))
    end
    function grad!(storage, x)
        storage .= 0
        for (idx, tensor_idx) in enumerate(selected_indices)
            storage[tensor_idx] = weights[idx] * (x[tensor_idx] - tensor_truth[tensor_idx]) / length(selected_indices)
        end
    end
    return f, grad!
end

"""
Build the Hessian for the completion problem
"""
function build_tensor_completion_hessian(tensor_truth, selected_indices, weights=ones(length(selected_indices)))
    n = length(selected_indices)
    d = zeros(length(tensor_truth))
    d[selected_indices] .= weights / n
    return Diagonal(d)
end

struct Scheduler{T}
    start_time::Int
    scaling_factor::T
    max_interval::Int
    current_interval::Base.RefValue{Int}
    last_solve_counter::Base.RefValue{Int}
end

Scheduler(; start_time=20, scaling_factor=1.5, max_interval=1000) =
    Scheduler(start_time, scaling_factor, max_interval, Ref(start_time), Ref(0))

function FrankWolfe.should_solve_lp(as::FrankWolfe.ActiveSetQuadraticLinearSolve, scheduler::Scheduler)
    if length(as) <= 10
        return false
    end
    if as.counter[] - scheduler.last_solve_counter[] >= scheduler.current_interval[]
        scheduler.last_solve_counter[] = as.counter[]
        scheduler.current_interval[] = min(
            round(Int, scheduler.scaling_factor * scheduler.current_interval[]),
            scheduler.max_interval,
        )
        return true
    end
    return false
end
