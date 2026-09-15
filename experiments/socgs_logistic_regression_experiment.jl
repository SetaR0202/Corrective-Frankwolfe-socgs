using FrankWolfe
using LinearAlgebra
using FileIO,JLD2
using LogExpFunctions
import Random

include("../src/socgs_algorithm.jl")
include("../src/load_gisette.jl")

socgs_record_dir_name = "socgs_records"
if !isdir(socgs_record_dir_name)
    mkdir(socgs_record_dir_name)
end
@info "gisette logistic regression over ℓ1 ball experiments"
#########################################################################################################
###Loading data
#########################################################################################################
@info "Loading data"
@time y,Z = load_gisette(do_normalize = false)
n,m = size(Z)
λ = 1/m
function logistic_loss(x)
    return λ/2 * dot(x,x) + sum(logsumexp.(0.0,-y .* (Z'*x)))/m
end

y_z_prod = Z * Diagonal(y)
function logistic_grad!(storage,x)
    storage .= λ*x - y_z_prod * (1 ./ (1 .+ exp.(y .* (Z'*x))))/m
end

function build_quadratic_approximation!(Hx,x,gradient,fx, H_quadratic_active_set, b_quadratic_active_set,t)
    Zx = Z' * x
    scales = 1 ./ ((1 .+ exp.(y .* Zx)) .* (1 .+ exp.(-y .* Zx)))
    Zxs = Zx .* scales    
    ZsZm = Z * Diagonal(scales) * Z' / m 
    H_quadratic_active_set .= λ*I(n) + ZsZm
    b_quadratic_active_set .= gradient - ZsZm * x - λ*x 
    function f_quad_approx(p)
        return dot(gradient, p-x) + 1/(2*m) * dot((Z' * p - Zx).^2, scales) + λ/2 * dot(p-x,p-x)
    end 
    function grad_quad_approx!(storage,p)
        storage .= gradient + 1/m * Z * (Z' * p .* scales - Zxs) + λ * (p-x)
    end
    return f_quad_approx, grad_quad_approx!
end


#############################################################################################################
####Run parameters
#############################################################################################################

##General SOCGS parameters
socgs_max_iteration = 50 #[5, 100, 1000]
socgs_timeout = 3000.0
do_lazy = true
lazy_tolerance = 1.0
fw_step = FrankWolfe.BlendedPairwiseStep(do_lazy,lazy_tolerance)
lmo = FrankWolfe.LpNormBallLMO{1}(1.0)
x0 = zeros(n)
x0[1] = 1.0

##Quadratic corrections parameters
do_pvm_with_quadratic_active_set = true#true to do a QC false for corrective step without QC (do_wolfe is then ignored)
do_wolfe = true #true for QC-MNP false QC-LP
scaling_factor = 30

##PVM parameters
pvm_stop_name = "100IT" #["100IT", "200IT", "500IT","1000IT"] #number of inner step for each pvm step
if pvm_stop_name == "100IT"
        lb_estimator =  LowerBoundConstant(0.) 
        pvm_max_iteration = 100 
elseif pvm_stop_name == "200IT"
    lb_estimator =  LowerBoundConstant(0.) 
    pvm_max_iteration = 200  
elseif pvm_stop_name == "500IT"
    lb_estimator =  LowerBoundConstant(0.) 
    pvm_max_iteration = 500
elseif pvm_stop_name == "1000IT"
    lb_estimator =  LowerBoundConstant(0.) 
    pvm_max_iteration = 1000
else
    throw(ErrorException("Unknown PVM stop name: "*pvm_stop_name))
end

line_search_after_pvm_name = "lsOFF" #["lsOFF","lsSECA","lsAGNO", "lsGAGN"]
if line_search_after_pvm_name== "lsOFF"
    line_search_after_pvm = nothing
elseif line_search_after_pvm_name=="lsSECA"
    line_search_after_pvm = FrankWolfe.Secant()
elseif line_search_after_pvm_name=="lsAGNO"
    line_search_after_pvm = FrankWolfe.Agnostic()
elseif line_search_after_pvm_name =="lsGAGN"
    line_search_after_pvm = FrankWolfe.GeneralizedAgnostic()
else
    throw(ErrorException("Unknown line search after PVM stop name: "*line_search_after_pvm_name))
end

##Verbosity and recording trajectories
socgs_verbose = true
socgs_trajectory = true
pvm_verbose = true
pvm_trajectory = false


#############################################################################################################
####RUN
#############################################################################################################
@info "Warming up" 

_ = second_order_conditional_gradient_sliding(
    logistic_loss,
    logistic_grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    fw_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration= 1,
    print_iter=1, 
    trajectory=socgs_trajectory, 
    verbose=socgs_verbose, 
    verbose_pvm = pvm_verbose ,
    traj_data=[],
    timeout=socgs_timeout,
    line_search_after_pvm = line_search_after_pvm,
    pvm_max_iteration = pvm_max_iteration,
    pvm_with_quadratic_active_set = do_pvm_with_quadratic_active_set, 
    H_quadratic_active_set = zeros(n,n),
    b_quadratic_active_set = zeros(n),
    scaling_factor = scaling_factor,        
    pvm_trajectory = pvm_trajectory,
    do_wolfe = do_wolfe
);

@info "Starting run" 
res = second_order_conditional_gradient_sliding(
    logistic_loss,
    logistic_grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    fw_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration= socgs_max_iteration,
    print_iter=1, 
    trajectory=socgs_trajectory,
    verbose=socgs_verbose, 
    verbose_pvm = pvm_verbose,
    traj_data=[],
    timeout=socgs_timeout,
    line_search_after_pvm = line_search_after_pvm,
    pvm_max_iteration = pvm_max_iteration,
    pvm_with_quadratic_active_set = do_pvm_with_quadratic_active_set, 
    H_quadratic_active_set = zeros(n,n),
    b_quadratic_active_set = zeros(n),
    scaling_factor = scaling_factor,
    pvm_trajectory = pvm_trajectory,
    do_wolfe = do_wolfe
);

#############################################################################################################
####Saving trajectories
#############################################################################################################
wolfe_ONOFF = do_wolfe ? "wolfeON" : "wolfeOFF"
qc_ONFF = do_pvm_with_quadratic_active_set ? "qcON" : "qcOFF"
suffix = wolfe_ONOFF * "_" * qc_ONFF *"_"*pvm_stop_name*"_"*line_search_after_pvm_name 
data = (traj = res.traj_data,)
save("socgs_records/socgs_gisette_it"*string(socgs_max_iteration)*"_timeout"*string(Int(round(socgs_timeout)))*"_"*suffix*".jld2","data",data)
    


