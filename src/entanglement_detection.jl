using Printf
using FrankWolfe
using Ket
import LinearAlgebra as LA
using Random
const syev_switch = 5

mutable struct BlasWorkspace{T}
    d::Int
    m::Base.RefValue{LA.BlasInt} # only for syevr
    W::Vector{T}
    Z::Matrix{Complex{T}}        # only for syevr
    isuppz::Vector{LA.BlasInt}   # only for syevr
    work::Vector{Complex{T}}
    lwork::LA.BlasInt
    rwork::Vector{T}
    lrwork::LA.BlasInt           # only for syevr
    iwork::Vector{LA.BlasInt}    # only for syevr
    liwork::LA.BlasInt           # only for syevr
    info::Base.RefValue{LA.BlasInt}
end

function BlasWorkspace(::Type{Float64}, d::Int)
    if d ≤ syev_switch
        W = Vector{Float64}(undef, d)
        work = Vector{ComplexF64}(undef, 33d)
        lwork = LA.BlasInt(33d)
        rwork = Vector{Float64}(undef, 3d-2)
        info = Ref{LA.BlasInt}()
        # dummy values for syevr specific fields
        m = Ref{LA.BlasInt}()
        Z = Matrix{ComplexF64}(undef, 0, 0)
        isuppz = Vector{LA.BlasInt}(undef, 0)
        lrwork = LA.BlasInt(0)
        iwork = Vector{LA.BlasInt}(undef, 0)
        liwork = LA.BlasInt(0)
    else
        m = Ref{LA.BlasInt}()
        W = Vector{Float64}(undef, d)
        Z = Matrix{ComplexF64}(undef, d, d)
        isuppz = Vector{LA.BlasInt}(undef, 2d)
        work = Vector{ComplexF64}(undef, 33d)
        lwork = LA.BlasInt(33d)
        rwork = Vector{Float64}(undef, 24d)
        lrwork = LA.BlasInt(24d)
        iwork = Vector{LA.BlasInt}(undef, 10d)
        liwork = LA.BlasInt(10d)
        info = Ref{LA.BlasInt}()
    end
    return BlasWorkspace{Float64}(d, m, W, Z, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

function BlasWorkspace(::Type{T}, d::Int) where {T <: Real}
    m = Ref{LA.BlasInt}()
    W = Vector{T}(undef, 0)
    Z = Matrix{Complex{T}}(undef, 0, 0)
    isuppz = Vector{LA.BlasInt}(undef, 0)
    work = Vector{Complex{T}}(undef, 0)
    lwork = LA.BlasInt(0)
    rwork = Vector{T}(undef, 0)
    lrwork = LA.BlasInt(0)
    iwork = Vector{LA.BlasInt}(undef, 0)
    liwork = LA.BlasInt(0)
    info = Ref{LA.BlasInt}()
    return BlasWorkspace{T}(d, m, W, Z, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end


function _syev!(A::Matrix{ComplexF64}, ws::BlasWorkspace{Float64})
    ccall((LA.BLAS.@blasfunc(zheev_), Base.liblapack_name), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ptr{Float64}, Ptr{ComplexF64}, Ref{LA.BlasInt},
           Ptr{Float64}, Ptr{LA.BlasInt}, Clong, Clong),
          'V', 'U', ws.d, A, stride(A, 2), ws.W, ws.work, ws.lwork, ws.rwork, ws.info, 1, 1)
    return ws.W, A
end

function _syevr!(A::AbstractMatrix{ComplexF64}, ws::BlasWorkspace{Float64})
    ccall((LA.BLAS.@blasfunc(zheevr_), Base.liblapack_name), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ref{ComplexF64}, Ref{ComplexF64}, Ref{LA.BlasInt},
           Ref{LA.BlasInt}, Ref{ComplexF64}, Ptr{LA.BlasInt}, Ptr{Float64},
           Ptr{ComplexF64}, Ref{LA.BlasInt}, Ptr{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ptr{Float64}, Ref{LA.BlasInt}, Ptr{LA.BlasInt},
           Ref{LA.BlasInt}, Ptr{LA.BlasInt}, Clong, Clong, Clong),
          'V', 'I', 'U', ws.d, A, stride(A, 2), 0.0, 0.0, 1, 1, -1.0, ws.m, ws.W, ws.Z, ws.d,
          ws.isuppz, ws.work, ws.lwork, ws.rwork, ws.lrwork, ws.iwork, ws.liwork, ws.info, 1, 1, 1)
    return ws.W, ws.Z
end


abstract type SeparableLMO{T, N} <: FrankWolfe.LinearMinimizationOracle end

"""
    Workspace{T, N}

Structure for initial pre-allocation of performance-critical functions.
"""
struct Workspace{T, N}
    pure_kets::NTuple{N, Vector{Complex{T}}} # vectors of pure product states
    pure_tensors::NTuple{N, Vector{T}} # tensors of pure product states
    reduced_setdiffs::NTuple{N, Vector{Int}} # used in _reduced_tensor!
    reduced_tensors::NTuple{N, Vector{T}} # tensors of reduced density matrices of one qudit
    reduced_matrices::NTuple{N, Matrix{Complex{T}}} # reduced density matrices of one qudit
    blas_workspaces::NTuple{N, BlasWorkspace{T}}
end

function Workspace{T, N}(dims::NTuple{N, Int}) where {T <: Real, N}
    pure_kets = ntuple(n -> Vector{Complex{T}}(undef, dims[n]), Val(N))
    pure_tensors = ntuple(n -> Vector{T}(undef, dims[n]^2), Val(N))
    reduced_setdiffs = ntuple(n -> setdiff(1:N, n), Val(N))
    reduced_tensors = ntuple(n -> Vector{T}(undef, dims[n]^2), Val(N))
    reduced_matrices = ntuple(n -> Matrix{Complex{T}}(undef, dims[n], dims[n]), Val(N))
    blas_workspaces = ntuple(n -> BlasWorkspace(T, dims[n]), Val(N))

    return Workspace{T, N}(pure_kets, pure_tensors, reduced_setdiffs, reduced_tensors, reduced_matrices, blas_workspaces)
end

struct Fwdata 
    fw_iter::Vector{Int}
    fw_time::Vector{Float64}
    lmo_counts::Vector{Int}
end

function Fwdata()
    fw_iter = [1]
    fw_time = [0.0]
    lmo_counts = [0]
    return Fwdata(fw_iter, fw_time, lmo_counts)
end


"""
    _eigmin!(ket::Vector, matrix::Matrix)

Computes the minimal real eigenvalue and updates `ket` in place
The variable `matrix` of size d × d also gets overwritten.
For BLAS-compatible types, uses `LAPACK.syev!` for d ≤ 5 and `LAPACK.syevr!` otherwise.
For other types, falls back to `eigen!`.
"""
function _eigmin!(ket::Vector{ComplexF64}, matrix::Matrix{ComplexF64}, ws::BlasWorkspace{Float64})
    λ, X = ws.d ≤ syev_switch ? _syev!(matrix, ws) : _syevr!(matrix, ws)
    ket .= view(X, :, 1)
    return λ[1]
end

"""
    AlternatingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}} <: SeparableLMO{T, N}

`AlternatingSeparableLMO` implements `compute_extreme_point(lmo, direction)` which returns a pure product state used in Frank-Wolfe algorithms.
The method used is an alternating algorithm starting from random pure states on each party and alternatively optimizing each reduced state via an eigendecomposition.

Type parameters:
- `T`: element type of the correlation tensor
- `N`: number of parties
- `MB`: type of the matrix basis

Fields:
- `dims`: dimensions of the reduced state on each party
- `matrix_basis`: matrix basis of the correlation tensor
- `max_iter`: maximum number of alternation steps
- `threshold`: threshold to stop the alternation
- `nb`: number of random rounds to find the possible global optimal
- `workspace`: contains fields pre-allocated performance-critical functions
- `tmp`: temporary vector for fast scalar products of bipartite tensors
"""
struct AlternatingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}} <: SeparableLMO{T, N}
    dims::NTuple{N, Int}
    matrix_basis::NTuple{N, Vector{MB}}
    max_iter::Int
    threshold::T
    nb::Int
    parallelism::Bool
    workspaces::Vector{Workspace{T, N}}
    fwdata::Fwdata
    tmp::Vector{T}
end

function AlternatingSeparableLMO(
    ::Type{T},
    dims::NTuple{N, Int};
    matrix_basis = _gellmann(Complex{T}, dims),
    max_iter = 10^2,
    threshold = Base.rtoldefault(T),
    nb = 20,
    parallelism = false,
    verbose = 0,
    kwargs...
) where {T <: Real, N}
    MB = typeof(matrix_basis[1][1])
    if verbose > 0
        println("Quantum state structure: ", N, "-partite with local dimensions ", dims)
        println("Device numerical accuracy is ", threshold)
        println("Data is stored as ", typeof(matrix_basis[1][1]))
        if parallelism
            println("Parallelism is enabled with ", Threads.nthreads(), " threads.")
        else
            println("Parallelism is disabled.")
        end
    end
    if parallelism
        prod(dims) < 16 && @warn "The system dimension is small, parallelism may not be effective."
        LA.BLAS.set_num_threads(1)
        workspaces = [Workspace{T, N}(dims) for _ in 1:Threads.nthreads()]
    else
        workspaces = [Workspace{T, N}(dims)]
    end
    fwdata = Fwdata()
    tmp = N == 2 ? Vector{T}(undef, dims[1]^2) : T[]
    return AlternatingSeparableLMO{T, N, MB}(dims, matrix_basis, max_iter, threshold, nb, parallelism, workspaces, fwdata, tmp)
end

function AlternatingCore(work::Workspace{T, N}, lmo::AlternatingSeparableLMO{T, N}, dir::AbstractArray{T, N}) where {T <: Real, N}
    for n in 1:N
        Random.randn!(work.pure_kets[n])
        LA.normalize!(work.pure_kets[n])
        _correlation_tensor_ket!(work.pure_tensors[n], work.pure_kets[n], lmo.matrix_basis[n])
    end

    obj = typemax(T)
    obj_last = typemax(T)
    tensors = ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N))
    for _ in 1:lmo.max_iter
        obj = zero(T)
        for n in 1:N
            _reduced_tensor!(work.reduced_tensors[n], work.pure_tensors, dir, n, work.reduced_setdiffs[n])
            _density_matrix!(work.reduced_matrices[n], work.reduced_tensors[n], lmo.matrix_basis[n])
            obj = _eigmin!(work.pure_kets[n], work.reduced_matrices[n], work.blas_workspaces[n])
            _correlation_tensor_ket!(work.pure_tensors[n], work.pure_kets[n], lmo.matrix_basis[n])
        end

        if obj_last - obj ≤ lmo.threshold
            break
        end

        obj_last = obj
    end
    for n in 1:N
        tensors[n] .= work.pure_tensors[n]
    end
    return (tensors = tensors, obj = obj)
end

function FrankWolfe.compute_extreme_point(lmo::AlternatingSeparableLMO{T, N}, dir::AbstractArray{T, N}; kwargs...) where {T <: Real, N}
    lmo.fwdata.lmo_counts[1] += 1
    if lmo.parallelism 
        tensors = [ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N)) for _ in 1:lmo.nb]
        objs = [typemax(T) for _ in 1:lmo.nb]
        threaded_foreach(lmo.nb) do tid, task
            tensors[task], objs[task] = AlternatingCore(lmo.workspaces[tid], lmo, dir)
        end
        idx = argmin(objs) # find the best pure state
        return PureState{T, N, typeof(lmo)}(tensors[idx], objs[idx], lmo)
    else
        best_obj = typemax(T)
        best_pure_tensors = ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N))
        for _ in 1:lmo.nb
            tensor, obj = AlternatingCore(lmo.workspaces[1], lmo, dir)
            if obj < best_obj
                best_obj = obj
                for n in 1:N
                    best_pure_tensors[n] .= tensor[n]
                end
            end
        end
        return PureState{T, N, typeof(lmo)}(best_pure_tensors, best_obj, lmo)
    end
end


"""
    correlation_tensor(ρ::Matrix{T}, dims::NTuple{N, Int})

Convert a dimension-(a)symmetry density matrix `ρ` to a correlation tensor Array{T, N}, with subspace dimensions `dims`.
"""
function correlation_tensor(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, matrix_basis = _gellmann(CT, dims)) where {CT <: Number, N}
    @assert size(ρ) == (prod(dims), prod(dims)) "Density matrix size is not compatible with the given dimensions."
    T = float(real(CT))
    C = Array{T, N}(undef, dims .^ 2)
    _correlation_tensor!(C, ρ, matrix_basis)
    return C
end
export correlation_tensor

function _correlation_tensor!(tensor::Array{T, N}, ρ::AbstractMatrix{Complex{T}}, matrix_basis::NTuple{N, Vector{MB}}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}, N}
    dims2 = collect(length.(matrix_basis))
    vi = ones(Int, N)
    for i in 0:prod(dims2)-1
        tensor[vi...] = real(LA.dot(kron((matrix_basis[ni][di] for (ni, di) in enumerate(vi))...), ρ))
        _update_odometer!(vi, dims2)
    end
    return tensor
end

# copied from Ket, but changed the convention to start from 1 as we want indices
function _update_odometer!(ind::AbstractVector{<:Integer}, base::AbstractVector{<:Integer})
    ind[1] += 1
    d = length(ind)
    @inbounds for i in 1:d
        if ind[i] > base[i]
            ind[i] = 1
            i < d ? ind[i+1] += 1 : return
        else
            return
        end
    end
end

"""
    _correlation_tensor_ket!(tensor::Vector{T}, φ::Vector{Complex{T}}, matrix_basis)

Convert a ket `φ` to a correlation tensor Vector{T}, with same subspace dimension `d`.
"""
function _correlation_tensor_ket!(tensor::Vector{T}, φ::AbstractVector{Complex{T}}, matrix_basis::Vector{MB}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    for i in eachindex(matrix_basis)
        tensor[i] = real(LA.dot(φ, LA.Hermitian(matrix_basis[i]), φ))
    end
    return tensor
end

"""
    _reduced_tensor!(tensor::Vector{T}, pure_tensors::NTuple{N, Vector{T}}, dir::Array{T, N}, j::Int, s::Vector{Int} = setdiff(1:N, j)) where {T <: Real, N}

Computes the correlation tensor of the `j`-th subsystem (the tensor-version of the partial trace).
When N=2, j=1, computes ⟨ϕ2|dir|ϕ2⟩ for dir ∈ H₁ ⊗ H₂
"""
function _reduced_tensor!(tensor::Vector{T}, pure_tensors::NTuple{N, Vector{T}}, dir::AbstractArray{T, N}, j::Int, s::Vector{Int} = setdiff(1:N, j)) where {T <: Real, N}
    tensor .= 0
    for ind in CartesianIndices(dir)
        b = one(T)
        for i in s
            b *= pure_tensors[i][ind[i]]
        end
        tensor[ind[j]] += b * dir[ind]
    end
end




"""
    PureState{T, N} <: AbstractArray{T, N}

Represents a pure product state. Each subsystem is a pure state stored as a tensor PureState.tensors[n].
"""
struct PureState{T, N, LMO} <: AbstractArray{T, N}
    tensors::NTuple{N, Vector{T}} # correlation tensors of the individual parties
    obj::T # =<`tensors`,∇>, the minimal real eigenvalue of gradient direction
    lmo::LMO # gives access to tmp (for bipartite scalar product) and the matrix basis used
end

function PureState(x::PureState{T, N, LMO}) where {T, N, LMO}
    return PureState{T, N, LMO}(x.tensors, x.obj, x.lmo)
end

Base.IndexStyle(::Type{<:PureState}) = IndexCartesian()
Base.size(ps::PureState) = Tuple(length.(ps.tensors))

Base.@propagate_inbounds function Base.getindex(ps::PureState{T, 2}, x::Vararg{Int, 2}) where {T <: Real}
    @boundscheck (checkbounds(ps, x...))
    return @inbounds getindex(ps.tensors[1], x[1]) * getindex(ps.tensors[2], x[2])
end

Base.@propagate_inbounds function Base.getindex(ps::PureState{T, N}, x::Vararg{Int, N}) where {T <: Real, N}
    @boundscheck (checkbounds(ps, x...))
    return @inbounds prod(getindex(ps.tensors[n], x[n]) for n in 1:N)
end

FrankWolfe.fast_dot(A::Array, ps::PureState) = conj(FrankWolfe.fast_dot(ps, A))

function FrankWolfe.fast_dot(ps::PureState{T, 2}, A::Array{T, 2}) where {T <: Real}
    LA.mul!(ps.lmo.tmp, A, ps.tensors[2])
    return LA.dot(ps.tensors[1], ps.lmo.tmp)
end

function FrankWolfe.fast_dot(ps::PureState{T, N}, A::Array{T, N}) where {T <: Real} where {N}
    return LA.dot(ps, A)
end

function FrankWolfe.fast_dot(ps1::PureState{T, 2}, ps2::PureState{T, 2}) where {T <: Real}
    return LA.dot(ps1.tensors[1], ps2.tensors[1]) * LA.dot(ps1.tensors[2], ps2.tensors[2])
end

function FrankWolfe.fast_dot(ps1::PureState{T, N}, ps2::PureState{T, N}) where {T <: Real, N}
    return prod(LA.dot(ps1.tensors[n], ps2.tensors[n]) for n in 1:N)
end


"""
    function density_matrix(tensor::Vector{T}, dims::NTuple{N,Int}, matrix_basis = _gellmann(T, dims)) where {T <: Real} where {N}

Convert tensors of pure states to density matrix (for eigendecomposition).
"""
function density_matrix(tensor::AbstractArray{T, N}, dims::NTuple{N, Int}, matrix_basis = _gellmann(T, dims)) where {T <: Real, N}
    ρ = LA.Hermitian(Matrix{Complex{T}}(undef, prod(dims), prod(dims)))
    return _density_matrix!(ρ, tensor, matrix_basis)
end
export density_matrix

function density_matrix(ps::PureState{T, N}) where {T <: Real, N}
    dims = ps.lmo.dims
    ρ = LA.Hermitian(Matrix{Complex{T}}(undef, prod(dims), prod(dims)))
    return _density_matrix!(ρ, ps, ps.lmo.matrix_basis)
end

function _density_matrix!(ρ::Matrix{Complex{T}}, tensor::AbstractArray{T, 1}, matrix_basis::Vector{MB}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    ρ .= 0
    for i in eachindex(matrix_basis), j in eachindex(matrix_basis[i])
        ρ[j] += tensor[i] * matrix_basis[i][j]
    end
    ρ ./= 2
    return ρ
end

function _density_matrix!(ρ::LA.Hermitian, tensor::AbstractArray{T, 1}, matrix_basis::NTuple{1, Vector{MB}}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    _density_matrix!(parent(ρ), tensor, matrix_basis[1])
    return ρ
end

function _density_matrix!(ρ::LA.Hermitian, tensor::AbstractArray{T, N}, matrix_basis::NTuple{N, Vector{MB}}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}, N}
    data = parent(ρ)
    data .= 0
    dims2 = collect(length.(matrix_basis))
    vi = ones(Int, N)
    for i in 0:prod(dims2)-1
        # kron allocates a lot, but this is fine for the moment (no performance critical function)
        data .+= tensor[vi...] * kron((matrix_basis[ni][di] for (ni, di) in enumerate(vi))...)
        _update_odometer!(vi, dims2)
    end
    data ./= 2^N
    return ρ # TODO confirm this is correct
end

# Ket does not normalise gellmann the same way we do, the first element of tensor should be treated differently
function _gellmann(::Type{CT}, dims::NTuple{N, Int}) where {CT <: Number, N}
    T = float(real(CT))
    matrix_basis = Ket.gellmann.(Complex{T}, dims)
    for n in 1:N
        matrix_basis[n][1] .*= sqrt(T(2)) / sqrt(T(dims[n]))
    end
    return matrix_basis
end

function build_callback(trajectory_arr, epsilon, shortcut, shortcut_scale, noise_mixture, rp, Δp, C, Id, verbose, callback_iter)
    primal_prev = Inf
    noise_update = true
    noise_update_count = 0
    if verbose == 1 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("#Atoms", 7),
            )
    elseif verbose == 1 && noise_mixture
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("Noise", 10),
            lpad("#Atoms", 7),
            )
    elseif verbose == 2 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s   %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("Time (sec)", 10),
            lpad("#It/sec", 10),
            lpad("#Atoms", 7),
            lpad("#LMOs", 7)
            )
    elseif verbose == 2 && noise_mixture
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s    %s    %s   %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Primal_prev", 12),
            lpad("Dual gap", 10),
            lpad("Noise", 10),
            lpad("Time (sec)", 10),
            lpad("#It/sec", 10),
            lpad("#Atoms", 7),
            lpad("#LMOs", 7)
            )
    end
    function callback(state, args...)
        if length(args) > 0
            active_set = args[1]
            push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set), rp[]))
        else
            active_set = []
            push!(trajectory_arr, (FrankWolfe.callback_state(state)..., rp[]))
        end
        state.lmo.fwdata.fw_iter[1] = state.t
        state.lmo.fwdata.fw_time[1] = state.time

        if (mod(state.t, callback_iter) == 0 || noise_update)
            if verbose == 1 && noise_mixture == false
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    lpad(length(active_set), 7)
                    )
            elseif verbose == 1 && noise_mixture
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %.4e    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    rp[],
                    lpad(length(active_set), 7)
                    )
            elseif verbose == 2 && noise_mixture == false
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %.4e    %s   %s    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    state.time,
                    lpad(Printf.@sprintf("%.4e", state.t / state.time), 10),
                    lpad(length(active_set), 7),
                    lpad(state.lmo.fwdata.lmo_counts[1], 7)
                    )
            elseif verbose == 2 && noise_mixture
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %s    %.4e    %.4e    %.4e    %s   %s    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    lpad(Printf.@sprintf("%.4e", primal_prev), 10),
                    state.dual_gap,
                    rp[],
                    state.time,
                    lpad(Printf.@sprintf("%.4e", state.t / state.time), 10),
                    lpad(length(active_set), 7),
                    lpad(state.lmo.fwdata.lmo_counts[1], 7)
                    )
            end
        end
        noise_update = false

        if state.primal < epsilon # stop if the primal is small enough (main stopping criterion)
            verbose > 0 &&  @info "primal is small enough"
            return false
        end

        if noise_mixture && state.primal < primal_prev && state.primal / state.dual_gap > 1 + state.dual_gap * 10^4 # update the noise
            noise_update_count += 1
            if noise_update_count > (1 / Δp) ÷ 10
                noise_update = true
                noise_update_count = 0
            end
            rp[] += Δp
            primal_prev = state.primal
            if length(args) > 0
                FrankWolfe.update_active_set_quadratic!(active_set, -((1 - rp[]) * C + rp[] * Id))
            end
            return true
        end

        if noise_mixture && rp[] > 1 # stop if the noise is fully added
            verbose > 0 && @info "noise is fully added"
            return false
        end

        if !noise_mixture && shortcut && state.primal / state.dual_gap > shortcut_scale # when gap is large enough -> entangled, stop. (remove if we not use it)
            verbose > 0 && @info "shortcut"
            return false
        end

        return true # control when to stop
    end
    return callback
end

"""
    separable_distance(ρ::AbstractMatrix{CT}; dims, measure, fw_algorithm, kwargs...)
    separable_distance(C::Array{T, N}; matrix_basis, measure, fw_algorithm, kwargs...)

Computes the distance between the quantum density matrix `ρ` and the separable space under a specific `measure` via a specific `fw_algorithm`:
```
f(ρ) = min_{σ ∈ SEP} g(ρ,σ)
```
For the density matrix `ρ`, if the argument `dims` is omitted equally-sized subsystems are assumed, which is solving on the symmetry bipartite separable space.
For the correlation tensor `C`, if the argument `matrix_basis` is omitted, Gell-Mann matrix is assumed, which is Pauli basis for qubit systems.

The quantum state can also be given by a correlation tensor `C` corresponding to the experimental data from a set of (over-)completed `matrix_basis`.
If the argument `matrix_basis` is omitted the generalized Gell-Mann basis are assumed.

The `measure` g(ρ,σ) can be set as
- `"2-norm"`,
- [`"relative-entropy"`](https://arxiv.org/abs/quant-ph/9702027),
- [squared `"Bures metric"`](https://arxiv.org/abs/quant-ph/9707035).

The `fw_algorithm` can be used as
- `FrankWolfe.frank_wolfe`
- `FrankWolfe.lazified_conditional_gradient`
- `FrankWolfe.away_frank_wolfe`
- `FrankWolfe.blended_pairwise_conditional_gradient`

Returns a named tuple `(σ, v, primal)` with:
- `σ` the closest density matrix in the separable space
- `v` the closest pure separable state ket on the boundary of the separable space
- `primal` primal value f(x), the distance to the separable space
- `active_set` all the pure separable states, which combined to the closest separable state `σ`
- `lmo` the structure for related computation
"""
function separable_distance(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    C = correlation_tensor(ρ, lmo.dims, lmo.matrix_basis)
    x, v, primal, noise_level, active_set, lmo = separable_distance(C, lmo; measure, fw_algorithm, kwargs...)
    return (σ = density_matrix(x, lmo.dims, lmo.matrix_basis), v = v, primal = primal / 2^(N-1), noise_level = noise_level, active_set = active_set, lmo = lmo)
end
export separable_distance

function separable_distance(C::Array{T, N}, matrix_basis::NTuple{N, Vector{<:AbstractMatrix{Complex{T}}}}, lmo::LMO=AlternatingSeparableLMO(T, dims); measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Real, N, LMO <: SeparableLMO{T, N}}
    @assert length(bases) == N
    lmo = AlternatingSeparableLMO(T, size(C); matrix_basis, kwargs...)
    return separable_distance(C, lmo; measure, fw_algorithm, kwargs...)
end

function separable_distance(
    C::Array{T, N},
    lmo::SeparableLMO{T, N};
    noise_mixture::Bool = false,
    noise = nothing,
    noise_level::T = T(0),
    noise_atol::Real = 1e-3,
    measure::String = "2-norm",
    fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient,
    ini_sigma = Matrix{Complex{T}}(LA.I, prod(lmo.dims), prod(lmo.dims)) / prod(lmo.dims),
    ini_tensor = correlation_tensor(ini_sigma, lmo.dims),
    active_set = nothing,
    epsilon = 1e-6,
    lazy = true,
    max_iteration = 10^5,
    verbose = 0,
    callback_iter = 10^3,
    shortcut = false, # primal > 10 dual_gap stopping criterion
    shortcut_scale = 10,
    kwargs...
) where {T <: Real, N}
    #verbose >0 && lmo.parallelism && @info "The number of threads is $(Threads.nthreads())"

    # left for consistency between runs
    Random.seed!(0)
    
    if isnothing(noise)
        noise = similar(C)
        noise .= T(0)
        noise[1] = T(1)
    end
    rp = Ref(noise_level)
    dotCChalf = LA.dot(C, C) / 2
    dotIIhalf = LA.dot(noise, noise) / 2
    dotCI = LA.dot(C, noise)
    function f_noise(x, rp)
        return (1 - rp[])^2 * dotCChalf + rp[]^2 * dotIIhalf + rp[] * (1 - rp[]) * dotCI + LA.dot(x, x) / 2 - (1 - rp[])* LA.dot(C, x) - rp[] * LA.dot(noise, x)
    end
    function grad_noise!(storage, x, rp) # in-place gradient computation
        @. storage = x - ((1 - rp[]) * C + rp[] * noise)
    end
    f(x) = f_noise(x, rp)
    grad!(storage, x) = grad_noise!(storage, x, rp)

    if active_set === nothing
        x0 = FrankWolfe.compute_extreme_point(lmo, ini_tensor - C)
        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), x0)], LA.I, -C)
    elseif active_set isa FrankWolfe.ActiveSetQuadraticProductCaching
        FrankWolfe.update_active_set_quadratic!(active_set, -C)
        x0 = FrankWolfe.get_active_set_iterate(active_set)
    end

    trajectory_arr = []
    callback = build_callback(trajectory_arr, epsilon, shortcut, shortcut_scale, noise_mixture, rp, noise_atol, C, noise, verbose, callback_iter)

    if fw_algorithm in [FrankWolfe.frank_wolfe, FrankWolfe.lazified_conditional_gradient]
        x, v, primal, dual_gap, traj_data = fw_algorithm(
            f,
            grad!,
            lmo,
            x0;
            line_search = FrankWolfe.Shortstep(one(T)),
            epsilon = zero(T), # avoid standard stopping criterion for the dual gap
            max_iteration,
            callback,
            verbose = false,
            kwargs...
        )
        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), x)], LA.I, -C)
    else
        x, v, primal, dual_gap, traj_data, active_set = fw_algorithm(
            f,
            grad!,
            lmo,
            active_set;
            line_search = FrankWolfe.Shortstep(one(T)),
            epsilon = epsilon,# zero(T), # avoid standard stopping criterion for the dual gap
            max_iteration,
            callback,
            verbose = false,
            lazy,
            kwargs...
        )
    end

    # print last iteration
    if verbose == 1 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            lpad(length(active_set), 7)
            )
    elseif verbose == 1 && noise_mixture
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %.4e    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            rp[],
            lpad(length(active_set), 7)
            )
    elseif verbose == 2 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %.4e    %s   %s    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            lmo.fwdata.fw_time[1],
            lpad(Printf.@sprintf("%.4e", lmo.fwdata.fw_iter[1] / lmo.fwdata.fw_time[1]), 10),
            lpad(length(active_set), 7),
            lpad(lmo.fwdata.lmo_counts[1], 7)
            )
    elseif verbose == 2 && noise_mixture
        Printf.@printf(
            stdout,
            "%s    %.4e    %s    %.4e    %.4e    %.4e    %s   %s    %s\n",
            lpad("Last", 12),
            primal,
            lpad(Printf.@sprintf("%.4e", primal_prev), 10),
            dual_gap,
            rp[],
            lmo.fwdata.fw_time[1],
            lpad(Printf.@sprintf("%.4e", lmo.fwdata.fw_iter[1] / lmo.fwdata.fw_time[1]), 10),
            lpad(length(active_set), 7),
            lpad(lmo.fwdata.lmo_counts[1], 7)
            )
    end
    return (x = x, v = v, primal = primal, noise_level = rp[], active_set = active_set, lmo = lmo, traj_data = traj_data)
end
