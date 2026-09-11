using FileIO,JLD2
using Plots
using FrankWolfe

include(joinpath(pathof(FrankWolfe), "../../examples/plot_utils.jl"))

socgs_plots_dir_name = "socgs_plots"
if !isdir(socgs_plots_dir_name)
    mkdir(socgs_plots_dir_name)
end

#############################################################################################################
####Run parameters
#############################################################################################################
socgs_max_iteration = 5 #[5, 100, 1000]
socgs_timeout = 2000.0
pvm_stop_name = "100IT" #["100IT", "200IT", "500IT","1000IT"] #number of inner step for each pvm step
line_search_after_pvm_name = "lsOFF" #["lsOFF","lsSECA","lsAGNO", "lsGAGN"]

#############################################################################################################
####PLOT
#############################################################################################################

prefix = "socgs_gisette_it"*string(socgs_max_iteration)*"_timeout"*string(Int(round(socgs_timeout)))
suffix = pvm_stop_name*"_"*line_search_after_pvm_name 
full_names =
[
    (name=prefix*"_wolfeOFF_qcOFF_"*suffix, shortname="BPCG"),
    (name=prefix*"_wolfeON_qcON_"*suffix, shortname="QC-MNP"),
    (name=prefix*"_wolfeOFF_qcON_"*suffix, shortname="QC-LP")
]

save_filename= "socgs_plots/"*prefix*"_"*suffix*".pdf"

data =[]
label =[]

for exp in full_names
    loaded_data = FileIO.load("socgs_records/"*exp.name*".jld2","data")
    traj = loaded_data.traj
    push!(data,traj)
    push!(label,exp.shortname)
end


plot_trajectories(
    data,
    label,
    filename =save_filename,
    marker_shapes = [:circle, :star5, :diamond])
@info "Saved in "*save_filename
