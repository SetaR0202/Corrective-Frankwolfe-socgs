using JSON
using Plots
using FrankWolfe

include(joinpath(pathof(FrankWolfe), "../../examples/plot_utils.jl"))


all_files = filter(readdir(".")) do s
    startswith(s, "tensor_") && endswith(s, ".json")
end

# all_files = ["tensor_experiment_lazy_rank_5_radius_14.0_size_15_1.json", "tensor_experiment_lazy_rank_5_radius_14.0_size_15_2.json"]

for file in all_files
    fileinfo = JSON.parsefile(file)
    # needed to avoid a bug in display
    for (k, v) in fileinfo
        if occursin("traj_", k)
            v_primal = getindex.(v, 2) .+ 1e-12
            setindex!.(v, v_primal, 2)
            v_dual = getindex.(v, 4) .+ 1e-12
            setindex!.(v, v_dual, 4)
        end
    end
    try
        plot_trajectories(
            [fileinfo["traj_bpcg"], fileinfo["traj_ds_wolfe"], fileinfo["traj_ds"], fileinfo["traj_bcg"]],
            ["BPCG", "QC-MNP", "QC-LP", "BCG"],
            filename="tensor_plots/$(file)_plot.pdf",
            marker_shapes=[:o, :star5, :diamond, :+],
        )
    catch e
        @info e
        @info file
    end
end
