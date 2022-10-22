###

# FITS DETERMINISTIC NNS, LOADS CHAINS, PLOTS ALL MODELS

###

using LinearAlgebra: \, factorize
using SparseArrays: sparse
using DiffEqFlux: FastChain, FastDense, initial_params, sciml_train, sigmoid
using Flux: swish, ADAM
using Statistics: mean
using Interpolations: LinearInterpolation
using CSV: File
using DataFrames: DataFrame, filter, levels
using StatsPlots
using Turing
using Optim
Turing.setadbackend(:tracker)

#= include(abspath(joinpath("continuous_girder", "analysis", "Configs.jl")))
using .Configs: ANALYSIS_DIR, FEM_DIR, IJSSEL_DIR =#

include("Utils.jl")
using .Utils: linterp_even, relative

include("FEMGirders.jl")
using .FEMGirders: fem_general_single_girder, bending_moments

include("GirderSections.jl")
using .GirderSections: girder_sections

include("FieldTest.jl")
using .FieldTest: Truck, TruckName, RightTruck, LeftTruck

include("DataPlotting.jl")
using .DataUtils: loadsensordata_sciml,loadsensordata, Il_obj, sciml_pred_il_plot, create_prediction_error_plot,
    multi_scale, findall_signchange

meas_path2 = "real-world-case/data/measurements_processed.csv"


# ------------------------------------------------------
# SETTINGS & CONTROL
# ------------------------------------------------------

# number of neurons per layer
num_neuron = 20

sensor_position = 65
# steel elastic modulus (should be the same that was used to get stresses from strain measurements)
E = 210e6

#Load

#BSciML
using DelimitedFiles
chain_bsciml = read("real-world-case/H1489-38points-500p-2input.jls", Chains)
x_RHDHV = readdlm("real-world-case/data/x_RHDHV")
preds_RHDHV = zeros(length(x_RHDHV[:,1]),2,9)
for i = 1:9

    preds_RHDHV[:,:,i] = reshape(readdlm("real-world-case/data/RHDHV_sensor$i"),:,2)

end

# ------------------------------------------------------
# TRUCK LOADING
# ------------------------------------------------------
# Trucks used for loading

right_truck_center = 5.700 / 2 - 3.625 / 2
right_truck = Truck(RightTruck, right_truck_center)

left_truck_center = 5.700 / 2 + 3.625 / 2
left_truck = Truck(LeftTruck, left_truck_center)

# ------------------------------------------------------
# LOAD DATA: FUGRO MEASUREMENTS (mesurements.csv - processed)
# ------------------------------------------------------
max_elem_length = 1.8

# Store sensor information
sensor_info = Dict("H1_S" => 20.42, "H2_S" => 34.82, "H3_S" => 47.700, "H4_S" => 61.970,
    "H5_S" => 68.600, "H7_S" => 96.800, "H8_S" => 113.9, "H9_S" => 123.900, "H10_S" => 147.500)

# All sensor names, for prediction results
all_sensor_names = collect(keys(sensor_info)) |> sort! |> x -> circshift(x, -1)
all_sensor_positions = [sensor_info[sensor_name] for sensor_name in all_sensor_names]

# Data sensor name and position
data_sensor_names = ["H1_S","H4_S", "H8_S", "H9_S"]#, "H4_S", "H8_S", "H9_S"]

data_sensor_positions = [sensor_info[sensor_name] for sensor_name in data_sensor_names]

girder = girder_sections(
    max_elem_length=max_elem_length * 1e3,
    additional_node_positions=all_sensor_positions * 1e3,
)

# Scale to match the units in this file
support_xs = girder["support_xs"] / 1e3
node_xs = girder["node_xs"] / 1e3
elem_c_ys = girder["elem_c_ys"] / 1e3
elem_h_ys = girder["elem_h_ys"] / 1e3
elem_I_zs = girder["elem_I_zs"] / 1e12
elem_EIs = elem_I_zs .* E
elem_W_bottom = elem_I_zs ./ (elem_h_ys .- elem_c_ys)

num_meas_x = length(node_xs)
meas_xs = LinRange(support_xs[1], support_xs[end], num_meas_x)

# Assemble stiffness mx
Ks, nodes, idx_keep = fem_general_single_girder(node_xs, elem_EIs, support_xs)

# convert to a sparse mx and factorize for performance
spar_Ks = sparse(Ks)
fact_spar_Ks = factorize(spar_Ks)

# useful quantities for later
num_dof = size(Ks, 1)
num_node = size(nodes, 1)
data_sensor_nodes = [findall(sensor_position .== node_xs)[1] for sensor_position in data_sensor_positions]
all_sensor_nodes = [findall(sensor_position .== node_xs)[1] for sensor_position in all_sensor_positions]


# elastic sectional modulus at sensor location
W_sensor = elem_W_bottom[data_sensor_nodes]

# COMPUTE STRESS INFL. LINE FOR SENSOR

function stress_at_sensor_fugro(sensor_node, load, load_node, W_sensor)
    f = zeros(Float64, num_node * 2)
    f[load_node*2-1] = load
    fs = f[idx_keep]
    us = fact_spar_Ks \ fs
    u = zeros(Float64, num_node * 2)
    u[idx_keep] = us
    m = bending_moments(u, nodes, elem_EIs)
    m_sensor = m[sensor_node]
    stress = m_sensor / W_sensor
    return stress
end

f_y = 0.5
stress_il = [stress_at_sensor_fugro.(sensor_node, f_y, 1:num_node, elem_W_bottom[sensor_node]) for sensor_node in data_sensor_nodes]
all_stress_il = [stress_at_sensor_fugro.(sensor_node, f_y, 1:num_node, elem_W_bottom[sensor_node]) for sensor_node in all_sensor_nodes]

#Change xs for interpolations
# "_less" variables store measurements with less points for fitting.

corrected_node_xs = copy(node_xs[2:end-1])
node_xs = corrected_node_xs
node_xs_less = node_xs[1:5:end]
num_meas_x = length(node_xs)
meas_xs = LinRange(support_xs[1], support_xs[end], num_meas_x)
meas_xs_less = meas_xs[1:5:end]

# Correct stress ils
stress_il = [stress_il[i][2:end-1] for i in 1:length(stress_il)]
all_stress_il = [all_stress_il[i][2:end-1] for i in 1:length(all_stress_il)]
stress_il_less = [stress_il[i][1:5:end] for i = 1:length(stress_il)]
# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

all_r_meas_stress_il = Any[]
all_l_meas_stress_il = Any[]

for name in (all_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata_sciml(name, meas_path2, girder)
    push!(all_r_meas_stress_il, right_meas_stress_il)
    push!(all_l_meas_stress_il, left_meas_stress_il)
end

all_data_r_meas_stress_il = Any[]
all_data_l_meas_stress_il = Any[]

for name in (data_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata_sciml(name, meas_path2, girder)
    push!(all_data_r_meas_stress_il, right_meas_stress_il)
    push!(all_data_l_meas_stress_il, left_meas_stress_il)
end

all_data_r_meas_stress_il_less = [all_data_r_meas_stress_il[i][1:5:end]
                                  for i = 1:length(all_data_r_meas_stress_il)]
all_data_l_meas_stress_il_less = [all_data_l_meas_stress_il[i][1:5:end]
                                  for i = 1:length(all_data_l_meas_stress_il)]

# ------------------------------------------------------
# DEFINE NEURAL NETWORKS
# ------------------------------------------------------
# The neural network NN(x,z) is defined and the weights initialized.
tanh2(x) = 2.0 .* tanh(x)

NN2 = FastChain(
    FastDense(2, num_neuron, swish),
    FastDense(num_neuron, num_neuron, swish),
    FastDense(num_neuron, 1),
)
NN3 = FastChain(
    FastDense(3, 64, swish),
    FastDense(64, 64, swish),
    FastDense(64, 32, swish),
    FastDense(32, 1),
)

p_init2 = initial_params(NN2)
p_init3 = initial_params(NN3)

# For MAP
p_init_map = append!(p_sciml,rand(2))


# ------------------------------------------------------
# DEFINE LOSS FUNCTION
# ------------------------------------------------------
# Define callback to help visualizing the results
function callback(p, l)
    # Turn off for optimizing
    @show l
    false
end
function stress_il_from_truck_2(p, stress_il,nodes, il_xs; truck_name::TruckName)
    bridge_w = 5.700
    bridge_L = 295.322

    if truck_name == RightTruck
        truck = right_truck
    elseif truck_name == LeftTruck
        truck = left_truck
    else
        throw(ArgumentError("Unknown truck_name: $truck_name."))
    end

    right_wheel_forces = truck.wheel_forces[:, 1]
    left_wheel_forces = truck.wheel_forces[:, 2]
    right_wheel_x = truck.wheel_long_pos[:, 1]
    left_wheel_x = truck.wheel_long_pos[:, 2]
    right_wheel_z = truck.wheel_tvs_pos[1, 1]
    left_wheel_z = truck.wheel_tvs_pos[1, 2]

    lateral_load_distr(x, z) = NN2([x ./ bridge_L, z ./ bridge_w], p)[1]

    force_il_r = lateral_load_distr.(nodes, right_wheel_z)
    force_il_l = lateral_load_distr.(nodes, left_wheel_z)

    stress_il_r = stress_il .* force_il_r
    stress_il_l = stress_il .* force_il_l

    interp_stress_il_r = linterp_even(nodes, stress_il_r, 0.0)
    interp_stress_il_l = linterp_even(nodes, stress_il_l, 0.0)

    function stress_from_truck(x)
        truck_stress = sum(interp_stress_il_r.(right_wheel_x .+ x) .* right_wheel_forces) +
                       sum(interp_stress_il_l.(left_wheel_x .+ x) .* left_wheel_forces)
        return truck_stress
    end
    pred_stress_il = stress_from_truck.(il_xs)

    return pred_stress_il
end

function stress_il_from_truck_3(p, stress_il, nodes, il_xs, sensor_pos; truck_name::TruckName)
    bridge_w = 5.700
    bridge_L = 295.322
    last_sensor = last(all_sensor_positions)

    if truck_name == RightTruck
        truck = right_truck
    elseif truck_name == LeftTruck
        truck = left_truck
    else
        throw(ArgumentError("Unknown truck_name: $truck_name."))
    end

    right_wheel_forces = truck.wheel_forces[:, 1]
    left_wheel_forces = truck.wheel_forces[:, 2]
    right_wheel_x = truck.wheel_long_pos[:, 1]
    left_wheel_x = truck.wheel_long_pos[:, 2]
    right_wheel_z = truck.wheel_tvs_pos[1, 1]
    left_wheel_z = truck.wheel_tvs_pos[1, 2]

    lateral_load_distr(x, z, s) = NN3([x ./ bridge_L, z ./ bridge_w, s ./ last_sensor], p)[1]

    force_il_r = lateral_load_distr.(nodes, right_wheel_z, sensor_pos)
    force_il_l = lateral_load_distr.(nodes, left_wheel_z, sensor_pos)

    stress_il_r = stress_il .* force_il_r
    stress_il_l = stress_il .* force_il_l

    interp_stress_il_r = linterp_even(nodes, stress_il_r, 0.0)
    interp_stress_il_l = linterp_even(nodes, stress_il_l, 0.0)

    function stress_from_truck(x)
        truck_stress = sum(interp_stress_il_r.(right_wheel_x .+ x) .* right_wheel_forces) +
                       sum(interp_stress_il_l.(left_wheel_x .+ x) .* left_wheel_forces)
        return truck_stress
    end
    pred_stress_il = stress_from_truck.(il_xs)

    return pred_stress_il
end

function sciml_get_mse(p, _truck_name, sensor_positions, meas)
    mse = 0.0
    len = length(sensor_positions)

    if _truck_name == RightTruck
        for i = 1:len
            pred_il = stress_il_from_truck_2(p, stress_il_less[i],node_xs_less, meas; truck_name=_truck_name)
            mse += mean(abs2.(pred_il .- (all_data_r_meas_stress_il_less[i])))

        end
    elseif _truck_name == LeftTruck
        for i = 1:len
            pred_il = stress_il_from_truck_2(p, stress_il_less[i],node_xs_less, meas; truck_name=_truck_name)
            mse += mean(abs2.(pred_il .- (all_data_l_meas_stress_il_less[i])))
        end
    end

    return mse / len
end

function loss_m(p, x) #monotonicity

    l = 0.0
    bridge_w = 5.700
    bridge_L = 295.322

    #create z grid:
    zr = LinRange(
        right_truck.wheel_tvs_pos[1, 1] - 0.1,
        left_truck.wheel_tvs_pos[1, 2] + 0.1,
        length(meas_xs_less))

    lateral_load_distr(x, z) = NN2([x ./ bridge_L, z ./ bridge_w], p)[1]

    for i = 1:length(x)
        for j = 2:length(zr)
            lld_diff = lateral_load_distr(x[i],zr[j]) - lateral_load_distr(x[i],zr[j-1])
            if lld_diff >= 0
                l += (lateral_load_distr(x[i],zr[j]))
            else
                l += 0.0
            end
        end
    end
    return l
end

loss_m(p_init2,meas_xs_less)

function loss_MAP(p,x,stress_il) #map

    l_map = 0.0
    num_param = length(initial_params(NN3))
    alpha = 0.09
    sig = sqrt(1.0 / alpha)

    Prior = MvNormal(zeros(num_param), sig .* ones(num_param))
    prior = logpdf(Prior,p[1:end-2]) + logpdf(truncated(Normal(0,10000),0,Inf),1000*p[end-1])
            + logpdf(truncated(Normal(0,10000),0,Inf),1000*p[end])

    for i = 1:length(stress_il)

        L_right = MvNormal(stress_il_from_truck_3(p[1:end-2],stress_il[i],node_xs_less,x,data_sensor_positions[i],truck_name = RightTruck),
        sqrt.((stress_il_from_truck_3(p[1:end-2],stress_il[i],node_xs_less,x,data_sensor_positions[i],truck_name = RightTruck)*0.1*p[end-1]).^2 .+ (1000*p[end])^2))
        l_right = logpdf(L_right,all_data_r_meas_stress_il_less[i]) / length(stress_il)
        L_left = MvNormal(stress_il_from_truck_3(p[1:end-2],stress_il[i],node_xs_less,x,data_sensor_positions[i],truck_name = LeftTruck),
        sqrt.((stress_il_from_truck_3(p[1:end-2],stress_il[i],node_xs_less,x,data_sensor_positions[i],truck_name = LeftTruck)*0.1*p[end-1]).^2 .+ (1000*p[end])^2))
        l_left = logpdf(L_left,all_data_l_meas_stress_il_less[i]) / length(stress_il)

        l_map += l_right + l_left

    end

    return - (l_map + prior)

end
loss_MAP(p_init_map,meas_xs_less,stress_il_less)


function loss(p) # choose mse, mle or map

    #monotonicity
    #l_m = loss_m(p,meas_xs_less)

    #mle
    #l_mle = loss_MLE(p,meas_xs_less,stress_il_less)

    #map
    #l_map = loss_MAP(p,meas_xs_less,stress_il_less)

    # mse
    mse_right = sciml_get_mse(p, RightTruck, data_sensor_positions, meas_xs_less) * 1e-6 # MPa^2
    mse_left = sciml_get_mse(p, LeftTruck, data_sensor_positions, meas_xs_less) * 1e-6 # MPa^2
    mse = mse_right + mse_left

    #total. Adjust w accordingly
    w = 2
    total_loss = mse #+ w *l_m

    return total_loss
end

loss(p_init2)

# ------------------------------------------------------
# FIT NEURAL NETWORK
# ------------------------------------------------------
max_iters = 200
@time res = sciml_train(loss, p_init2,
    ADAM(0.05), maxiters=max_iters, save_best=true, cb=callback)

# refine with a lower learning rate
res = sciml_train(loss, res.minimizer,
    ADAM(0.003), maxiters=max_iters, save_best=true, cb=callback)

p_sciml = res.minimizer
#p_scimlmap = res.minimizer

right_pred_stress_il = stress_il_from_truck_2(p_sciml, all_stress_il[7],node_xs, meas_xs; truck_name=RightTruck)
left_pred_stress_il = stress_il_from_truck_2(p_sciml, all_stress_il[7],node_xs, meas_xs; truck_name=LeftTruck)

plot(meas_xs,right_pred_stress_il)
plot!(meas_xs,left_pred_stress_il)
plot!(meas_xs,all_data_r_meas_stress_il[3])
plot!(meas_xs,all_data_l_meas_stress_il[3])


MAP_pred_mean_l = zeros(length(meas_xs),length(all_sensor_positions))
MAP_pred_mean_r = zeros(length(meas_xs),length(all_sensor_positions))
for i = 1:length(all_sensor_positions)
    MAP_pred_mean_r[:,i] =  stress_il_from_truck_3(p_scimlmap,all_stress_il[i],node_xs,meas_xs,all_sensor_positions[i],truck_name = RightTruck)
    MAP_pred_mean_l[:,i] =  stress_il_from_truck_3(p_scimlmap,all_stress_il[i],node_xs,meas_xs,all_sensor_positions[i],truck_name = LeftTruck)
end

q_l = zeros(length(meas_xs),2,length(all_sensor_positions))
q_u = zeros(length(meas_xs),2,length(all_sensor_positions))

for i = 1:length(meas_xs)
    for j = 1:length(all_sensor_positions)
        d_r = Normal(stress_il_from_truck_3(p_scimlmap,all_stress_il[j],node_xs,meas_xs[i],all_sensor_positions[j],truck_name = RightTruck),
                sqrt.((stress_il_from_truck_3(p_scimlmap[1:end-2],all_stress_il[j],node_xs,meas_xs[i],all_sensor_positions[j],truck_name = RightTruck)*0.1*p_scimlmap[end-1]).^2 .+ (1000*p_scimlmap[end])^2))
        d_l = Normal(stress_il_from_truck_3(p_scimlmap,all_stress_il[j],node_xs,meas_xs[i],all_sensor_positions[j],truck_name = LeftTruck),
                sqrt.((stress_il_from_truck_3(p_scimlmap[1:end-2],all_stress_il[j],node_xs,meas_xs[i],all_sensor_positions[j],truck_name = LeftTruck)*0.1*p_scimlmap[end-1]).^2 .+ (1000*p_scimlmap[end])^2))
        q_u[i,1,j] = quantile(d_r,0.95)
        q_l[i,1,j] = quantile(d_r,0.05)
        q_u[i,2,j] = quantile(d_l,0.95)
        q_l[i,2,j] = quantile(d_l,0.05)
    end
end

plot(meas_xs,MAP_pred_mean_r[:,7])
plot!(meas_xs,all_r_meas_stress_il[7])

#RMSE

rmse_r = sqrt(mean(abs2.((1e-3 .* bsciml_pred_mean[1, :, 1, 7]) .- (1e-3 .* all_r_meas_stress_il[7]))))
rmse_l = sqrt(mean(abs2.((1e-3 .* bsciml_pred_mean[1, :, 2, 7]) .- (1e-3 .* all_l_meas_stress_il[7]))))

rmse = (rmse_l + rmse_r)/2

function plot_map(sensor_id)
        p1 =
        plot(meas_xs, 1e-3 .* MAP_pred_mean_r[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,1,sensor_id], fillrange=1e-3 .* q_u[:,1,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
        label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)
        #plot!(meas_xs, 0.2 .* all_stress_il[1], ls=:dash, w=2, label="Fe_model")

        p2 =
        plot(meas_xs, 1e-3 .*MAP_pred_mean_l[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,2,sensor_id], fillrange=1e-3 .* q_u[:,2,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
        label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)
        t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

        P = plot(t, p2, p1, layout = @layout([A{0.01h}; [B C]]), size = (1500, 500),#=  ylims = (-30,50), =#
            title = ["Test sensor : $(all_sensor_positions[sensor_id]) m " "Left truck lane" "Right truck lane"], left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#

    return P
end

plot_map(9)

# ------------------------------------------------------
# BSciML
# ------------------------------------------------------

# modify function to output only FE prediction
# faster to only pre compute this outisde BNN
function stress_il_from_truck_bnn(x_obs, il_xs, stress_il_fe; truck_name::TruckName)
    if truck_name == RightTruck
        truck = right_truck
    elseif truck_name == LeftTruck
        truck = left_truck
    else
        throw(ArgumentError("Unknown truck_name: $truck_name."))
    end

    right_wheel_forces = truck.wheel_forces[:, 1]
    left_wheel_forces = truck.wheel_forces[:, 2]
    right_wheel_x = truck.wheel_long_pos[:, 1]
    left_wheel_x = truck.wheel_long_pos[:, 2]

    interp_stress_il = linterp_even(il_xs, stress_il_fe, 0.0)


    function stress_from_truck_r(x)
        return sum(interp_stress_il.(right_wheel_x .+ x) .* right_wheel_forces)
    end
    function stress_from_truck_l(x)
        return sum(interp_stress_il.(left_wheel_x .+ x) .* left_wheel_forces)
    end

    pred_stress_il_r = stress_from_truck_r.(x_obs)
    pred_stress_il_l = stress_from_truck_r.(x_obs)

    return pred_stress_il_r, pred_stress_il_l
end

# Generate FE influence lines predictions

#For predictions
right_wheel_il_all = Array{Vector}(undef, length(all_stress_il))
left_wheel_il_all = Array{Vector}(undef, length(all_stress_il))
for i = 1:length(all_stress_il) #loop over data sensors

    (right_wheel_right_truck, left_wheel_right_truck) = stress_il_from_truck_bnn(meas_xs, node_xs, all_stress_il[i]; truck_name=RightTruck)
    (right_wheel_left_truck, left_wheel_left_truck) = stress_il_from_truck_bnn(meas_xs, node_xs, all_stress_il[i]; truck_name=LeftTruck)

    #Pass to matrix
    right_wheel_il_all[i] = [right_wheel_right_truck, right_wheel_left_truck]
    left_wheel_il_all[i] = [left_wheel_right_truck, left_wheel_left_truck]

end
num_param2 = length(initial_params(NN2))

_NN = FastChain(
    FastDense(3, 20, swish),
    FastDense(20, 20, swish),
    FastDense(20, 1),
)

num_param3 = length(initial_params(_NN))
alpha = 0.09
sig = sqrt(1.0 / alpha)

@model function bayesian_sciml_realworld_2(observations, x, Right_wheel_il, Left_wheel_il,
    ::Type{T}=Float64) where {T}

    bridge_w = 5.700
    bridge_L = 295.322

    #prior parameters
    θ ~ MvNormal(zeros(num_param2), sig .* ones(num_param2))
    std ~  InverseGamma(3, 4)

    #TLD definition and constants
    lateral_load_distr(x, z) = NN2([x ./ bridge_L, z ./ bridge_w], θ)[1]

    #= right_wheel_z = [right_truck.wheel_tvs_pos[1, 1], left_truck.wheel_tvs_pos[1, 1]]
    left_wheel_z = [right_truck.wheel_tvs_pos[1, 2], left_truck.wheel_tvs_pos[1, 2]] =#
    pred_il = Array{T}(undef, length(x), 2, length(Right_wheel_il))

    #= for i = 1:length(Right_wheel_il) #loops over sensors
        #Compute TLD and Fe_model * TLD
        for j = 1:2 #i loops over the two truck lanes

            TLD_r = lateral_load_distr.(x, right_wheel_z[j])
            TLD_l = lateral_load_distr.(x, left_wheel_z[j])

            pred_il[:, j, i] = (Right_wheel_il[i][j] .* TLD_r) + (Left_wheel_il[i][j] .* TLD_l)

        end
    end =#

    TLD_r = lateral_load_distr.(x, right_truck_center)
    TLD_l = lateral_load_distr.(x,left_truck_center)
    for i = 1:length(Right_wheel_il)
        pred_il[:,1,i] = TLD_r .* (Right_wheel_il[i][1] + Left_wheel_il[i][1])
        pred_il[:,2,i] = TLD_l .* (Right_wheel_il[i][2] + Left_wheel_il[i][2])
    end

    #Likelihood
    observations ~ MvNormal(pred_il[:], 100 * std)

end


@model function bayesian_sciml_realworld_3(observations, x, Right_wheel_il, Left_wheel_il,
    ::Type{T}=Float64) where {T}

    bridge_w = 5.700
    bridge_L = 295.322

    #prior parameters
    θ ~ MvNormal(zeros(num_param3), sig .* ones(num_param3))
    std ~ InverseGamma(3, 4)

    #TLD definition and constants
    lateral_load_distr(x, z, s) = _NN([x ./ bridge_L, z ./ bridge_w, s ./ last(all_sensor_positions)], θ)[1]

    #= right_wheel_z = [right_truck.wheel_tvs_pos[1, 1], left_truck.wheel_tvs_pos[1, 1]]
    left_wheel_z = [right_truck.wheel_tvs_pos[1, 2], left_truck.wheel_tvs_pos[1, 2]] =#
    pred_il = Array{T}(undef, length(x), 2, length(Right_wheel_il))

    #Necessary to dintinguish between inference
    #and prediction due to the different number
    #of sensors used in both cases.

    if ismissing(observations) == false #if inference
        #= for i = 1:length(Right_wheel_il) #loops over sensors
            #Compute TLD and Fe_model * TLD
            for j = 1:2 #i loops over the two truck lanes

                TLD_r = lateral_load_distr.(x, right_wheel_z[j], data_sensor_positions[i])
                TLD_l = lateral_load_distr.(x, left_wheel_z[j], data_sensor_positions[i])

                pred_il[:, j, i] = (Right_wheel_il[i][j] .* TLD_r) + (Left_wheel_il[i][j] .* TLD_l)

            end
        end =#

        for i = 1:length(Right_wheel_il)
            TLD_r = lateral_load_distr.(x, right_truck_center, data_sensor_positions[i])
            TLD_l = lateral_load_distr.(x,left_truck_center, data_sensor_positions[i])

            pred_il[:,1,i] = TLD_r .* (Right_wheel_il[i][1] + Left_wheel_il[i][1])
            pred_il[:,2,i] = TLD_l .* (Right_wheel_il[i][2] + Left_wheel_il[i][2])
        end

    else # if predicting
        #= for i = 1:length(Right_wheel_il) #loops over sensors
            #Compute TLD and Fe_model * TLD
            for j = 1:2 #i loops over the two truck lanes

                TLD_r = lateral_load_distr.(x, right_wheel_z[j], all_sensor_positions[i])
                TLD_l = lateral_load_distr.(x, left_wheel_z[j], all_sensor_positions[i])

                pred_il[:, j, i] = (Right_wheel_il[i][j] .* TLD_r) + (Left_wheel_il[i][j] .* TLD_l)

            end
        end =#

        for i = 1:length(Right_wheel_il)
            TLD_r = lateral_load_distr.(x, right_truck_center, all_sensor_positions[i])
            TLD_l = lateral_load_distr.(x,left_truck_center, all_sensor_positions[i])

            pred_il[:,1,i] = TLD_r .* (Right_wheel_il[i][1] + Left_wheel_il[i][1])
            pred_il[:,2,i] = TLD_l .* (Right_wheel_il[i][2] + Left_wheel_il[i][2])
        end
    end

    #Likelihood
    observations ~ MvNormal(pred_il[:],100 * std)

end

test_model_bsciml_realworld = bayesian_sciml_realworld_2(missing, meas_xs, right_wheel_il_all, left_wheel_il_all);
bsciml_pred = reshape(Array(predict(test_model_bsciml_realworld, chain_bsciml)), (1000, length(meas_xs), 2, 9))
bsciml_pred_mean = mean(bsciml_pred, dims=1)
bsciml_median = median(bsciml_pred, dims = 1)
plot(meas_xs,bsciml_pred_mean[1,:,1,1])
plot!(meas_xs,all_data_r_meas_stress_il[1])
plot!(x_RHDHV,1000 .* preds_RHDHV[:,2,1])
nanquantile(x, q) = quantile(filter(!isnan, x), q)

function compare_with_RHDHV_preds(sensor_id)


    Upper_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(meas_xs)]

    begin
        gr()
        p1 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_r, fillrange=1e-3 .* Upper_bsciml_r,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="", yaxis="",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        #= plot(meas_xs, 1e-3 .* MAP_pred_mean_r[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,1,sensor_id], fillrange=1e-3 .* q_u[:,1,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14) =#
        plot!(x_RHDHV, preds_RHDHV[:,2,sensor_id], w=2, label="RHDHV", color="red",xlabel="Longitudinal direction, x [m]")
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)
        #plot!(meas_xs, 0.2 .* all_stress_il[1], ls=:dash, w=2, label="Fe_model")

        p2 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_l, fillrange=1e-3 .* Upper_bsciml_l,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        #= plot(meas_xs, 1e-3 .*MAP_pred_mean_l[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,2,sensor_id], fillrange=1e-3 .* q_u[:,2,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14) =#
        plot!(x_RHDHV, preds_RHDHV[:,1,sensor_id], w=2, label="RHDHV", color="red",xlabel="Longitudinal direction, x [m]")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)

        t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

        P = plot(t, p2, p1, link = :y, layout = @layout([A{0.01h}; [B C]]), size = (1500, 500),#=  ylims = (-30,50), =#
            title = ["Test sensor : $(all_sensor_positions[sensor_id]) m " "Left truck lane" "Right truck lane"], left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    end

    return P
end

compare_with_RHDHV_preds(8)

begin
    plotlyjs()
    l = @layout [grid(3,3)]
    plots = []
    for i = 1:9
        if i == 2 || i == 3 || i ==5 || i==6
            p =
            plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, i], lw=3, label="BSciML, right", color="blue",
                title="", xticks = false,yticks = false,legend = false)
            plot!(x_RHDHV, preds_RHDHV[:,2,i], w=2, label="RHDHV, right", color="blue",
                xlabel="", ls =:dash, xticks = false,yticks = false,legend = false)
            plot!(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, i], lw=3, label="BSciML, left", color="red",
                title="", xticks = false,yticks = false,legend = false)
            plot!(x_RHDHV, preds_RHDHV[:,1,i], w=2, label="RHDHV, left", color="red",
                xlabel="", ls =:dash, xticks = false,yticks = false,legend = false)
            plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[i],
                label="Measurements", w=2, color="green", xticks = false,yticks = false)
            plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[i],
                label="", w=2, color="green", xticks = false,yticks = false, legend = false)
            vline!(support_xs, ls=:dash, label=false, color="black")
            hline!([0], label=false, color="black", xticks = false,yticks = false)
            scatter!([all_sensor_positions[i]], [0], label=false, xticks = false,yticks = false, legend = false)

        elseif i == 8 || i == 9

            p =
            plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, i], lw=3, label="BSciML, right", color="blue",
                title="",yticks = false, xlabel = "Longitudinal direction, x [m]")
            plot!(x_RHDHV, preds_RHDHV[:,2,i], w=2, label="RHDHV, right", color="blue",
                xlabel="", ls =:dash,yticks = false)
            plot!(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, i], lw=3, label="BSciML, left", color="red",
                title="",yticks = false)
            plot!(x_RHDHV, preds_RHDHV[:,1,i], w=2, label="RHDHV, left", color="red",
                xlabel="", ls =:dash,yticks = false)
            plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[i],
                label="Measurements", w=2, color="green",yticks = false)
            plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[i],
                label="", w=2, color="green",yticks = false, legend = false)
            vline!(support_xs, ls=:dash, label=false, color="black")
            hline!([0], label=false, color="black",yticks = false)
            scatter!([all_sensor_positions[i]], [0], label=false,yticks = false, legend = false)

        elseif i == 1 || i == 4
            p =
            plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, i], lw=3, label="BSciML, right", color="blue",
                title="", xticks = false,ylabel = "Stress [MPa]")
            plot!(x_RHDHV, preds_RHDHV[:,2,i], w=2, label="RHDHV, right", color="blue",
                xlabel="", ls =:dash, xticks = false)
            plot!(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, i], lw=3, label="BSciML, left", color="red",
                title="", xticks = false)
            plot!(x_RHDHV, preds_RHDHV[:,1,i], w=2, label="RHDHV, left", color="red",
                xlabel="", ls =:dash, xticks = false)
            plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[i],
                label="Measurements", w=2, color="green", xticks = false)
            plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[i],
                label="", w=2, color="green", xticks = false, legend = false)
            vline!(support_xs, ls=:dash, label=false, color="black", xticks = false)
            hline!([0], label=false, color="black", xticks = false)
            scatter!([all_sensor_positions[i]], [0], label=false, xticks = false, legend = false)

        else

            p =
            plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, i], lw=3, label="BSciML, right", color="blue",
                title="",ylabel = "Stress [MPa]")
            plot!(x_RHDHV, preds_RHDHV[:,2,i], w=2, label="RHDHV, right", color="blue",
                xlabel="", ls =:dash)
            plot!(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, i], lw=3, label="BSciML, left", color="red",
                title="")
            plot!(x_RHDHV, preds_RHDHV[:,1,i], w=2, label="RHDHV, left", color="red",
                xlabel="", ls =:dash)
            plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[i],
                label="Measurements", w=2, color="green")
            plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[i],
                label="Measurements", w=2, color="green")
            vline!(support_xs, ls=:dash, label=false, color="black")
            hline!([0], label=false, color="black")
            scatter!([all_sensor_positions[i]], [0], label=false)

        end
        push!(plots,p)

    end


    p9 =
    plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, 9], lw=3, label="BSciML, right", color="blue",
        title="")
    plot!(x_RHDHV, preds_RHDHV[:,2,9], w=2, label="RHDHV, right", color="blue",
        xlabel="", ls =:dash)
    plot!(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, 9], lw=3, label="BSciML, left", color="red",
        title="")
    plot!(x_RHDHV, preds_RHDHV[:,1,9], w=2, label="RHDHV, left", color="red",
        xlabel="", ls =:dash)
    plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[9],
        label="Measurements", w=2, color="green")
    plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[9],
        label="", w=2, color="green")
    vline!(support_xs, ls=:dash, label=false, color="black")
    hline!([0], label=false, color="black")
    scatter!([all_sensor_positions[9]], [0], label=false,xlabel = "Longitudinal direction, x [m]")

    #t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)
    #= bottom = plot(xlabel = "Longitudinal direction, x [m]", grid = false, showaxis = false, xticks = false, yticks = false,legend = false)
    left = plot(ylabel = "Stress [MPa]", grid = false, showaxis = false, xticks = false, yticks = false,legend = false) =#
    plot(plots[1],plots[2],plots[3],plots[4],plots[5],plots[6],plots[7],plots[8],p9,
    layout = l, title = "",left_margin = 4Plots.mm,bottom_margin = 4Plots.mm, size = (900,900),xguidefontsize=14, yguidefontsize=14,
    link = :y)



end

###########################################################

# Pure NN and BNN

###########################################################

z_truck = [right_truck_center,left_truck_center]
girder_dd = girder_sections(
    max_elem_length= 2.0 * 1e3,
    additional_node_positions=data_sensor_positions * 1e3,
)

# Scale to match the units in this file
support_xs_dd = girder_dd["support_xs"] / 1e3
node_xs_dd = girder_dd["node_xs"] / 1e3

# Interpolate the measurements over a shared x grid
num_meas_x_dd = length(node_xs_dd)
meas_xs_dd = LinRange(support_xs_dd[1], support_xs_dd[end], num_meas_x_dd)

all_r_meas_stress_il_dd = Any[]
all_l_meas_stress_il_dd = Any[]

for name in (data_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata(name, meas_path2, girder_dd)
    push!(all_r_meas_stress_il_dd, right_meas_stress_il)
    push!(all_l_meas_stress_il_dd, left_meas_stress_il)
end

meas_stress_il_matrix = zeros(length(node_xs_dd),2,length(all_l_meas_stress_il_dd))
for i = 1:length(all_r_meas_stress_il_dd)
    meas_stress_il_matrix[:,1,i] = all_r_meas_stress_il_dd[i]
    meas_stress_il_matrix[:,2,i] = all_l_meas_stress_il_dd[i]
end

NN = FastChain(
    FastDense(3, 20, swish),
    FastDense(20, 1),
)

p_init = initial_params(NN)
function nn_pred(x::AbstractFloat, z::AbstractFloat, sensor::AbstractFloat, p)
    y = NN([x, z, sensor], p)[1]
  return y
end

function dd_stress_il_from_truck(x, z, sensor_position, p)

    stress_scale = 50
    bridge_w = 5.700
    bridge_L = 295.322

    pred_stress_il = nn_pred(x / bridge_L , z / bridge_w, sensor_position / bridge_L, p)

    return pred_stress_il .* stress_scale
end


function dd_get_mse(x,z,sensor_positions,p)
    mse = 0.0
    len = length(sensor_positions)
    for i = 1:len
        for j = 1:length(z)
            mse += mean(abs2.(dd_stress_il_from_truck.(x, z[j], sensor_positions[i],p) .- meas_stress_il_matrix[:,j,i]))
        end
    end

    return mse / len
end


function loss(p)
    mse = dd_get_mse(node_xs_dd, z_truck, data_sensor_positions,[p]) * 1e-6 # MPa^2
    return mse
end
loss(p_init)

# ------------------------------------------------------
# FIT NEURAL NETWORK
# ------------------------------------------------------
max_iters = 2000
res = sciml_train(loss, p_init,
    ADAM(0.05), maxiters=max_iters, save_best=true, cb=callback)

# refine with a lower learning rate
res = sciml_train(loss, res.minimizer,
    ADAM(0.01), maxiters=max_iters, save_best=true, cb=callback)

p_dd = res.minimizer

@model function bayesian_dd_realworld(observations, x, z, sensor, ::Type{T} = Float64) where {T}

    #prior parameters
    θ ~ MvNormal(zeros(num_param), sig .* ones(num_param))
    std ~ InverseGamma(2, 1)

    #Loop over the chosen sensors and load lanes
    preds = Array{T}(undef, length(x), length(z), length(sensor))

    for i = 1:length(sensor)
        for j = 1:length(z)
            # call neural network prediction
            preds[:, j, i] = dd_stress_il_from_truck.(x, z[j], sensor[i], [θ])
        end
    end

    #Flatten pred
    preds_flat = preds[:]

    #Likelihood
    observations ~ MvNormal(preds_flat, std)

end


chain_bnn = read("RealWorld/Results/Document/H1489-bnn-33points-500p.jls", Chains)
NN = FastChain(
    FastDense(3, 20, swish),
    FastDense(20, 20, swish),
    FastDense(20, 1),
)
num_param = length(initial_params(NN))
test_model_bnn_realworld = bayesian_dd_realworld(missing, node_xs_dd, z_truck, all_sensor_positions);
bnn_pred = reshape(Array(predict(test_model_bnn_realworld,chain_bnn)),(1000,length(node_xs_dd),2,9))
bnn_pred_mean = mean(bnn_pred, dims = 1)


#PLOTS

function bsciml_prediction_2(sensor_id)

    right_pred_stress_il = stress_il_from_truck_2(p, all_stress_il[sensor_id],node_xs, meas_xs; truck_name=RightTruck)
    left_pred_stress_il = stress_il_from_truck_2(p, all_stress_il[sensor_id],node_xs, meas_xs; truck_name=LeftTruck)

    NN = FastChain(
        FastDense(3, 32, swish),
        FastDense(32, 1),
    )

    right_pred_stress_il_dd = dd_stress_il_from_truck.(node_xs_dd, z_truck[1], all_sensor_positions[sensor_id],[p_dd])
    left_pred_stress_il_dd = dd_stress_il_from_truck.(node_xs_dd, z_truck[2], all_sensor_positions[sensor_id],[p_dd])

    Upper_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bnn_r = [nanquantile(Array(bnn_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(node_xs_dd)]
    Lower_bnn_r = [nanquantile(Array(bnn_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(node_xs_dd)]

    Upper_bnn_l = [nanquantile(Array(bnn_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(node_xs_dd)]
    Lower_bnn_l = [nanquantile(Array(bnn_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(node_xs_dd)]


    begin
        gr()
        p1 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_r, fillrange=1e-3 .* Upper_bsciml_r,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(meas_xs, 1e-3 .* right_pred_stress_il, w=2, label="SciML", color="red")
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)
        #plot!(meas_xs, 0.2 .* all_stress_il[1], ls=:dash, w=2, label="Fe_model")

        p2 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_l, fillrange=1e-3 .* Upper_bsciml_l,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(meas_xs, 1e-3 .* left_pred_stress_il, w=2, label="SciML", color="red")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)

        p3 =
        plot(node_xs_dd, 1e-3 .* bnn_pred_mean[1, :, 2, sensor_id], lw=3, label="BNN", color="blue",
            title="")
        plot!(node_xs_dd, 1e-3 .* Lower_bnn_l, fillrange=1e-3 .* Upper_bnn_l,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(node_xs_dd, 1e-3 .* right_pred_stress_il_dd, w=2, label="NN", color="red")
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs_dd, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)


        p4 =
        plot(node_xs_dd, 1e-3 .* bnn_pred_mean[1, :, 1, sensor_id], lw=3, label="BNN", color="blue",
        title="")
        plot!(node_xs_dd, 1e-3 .* Lower_bnn_r, fillrange=1e-3 .* Upper_bnn_r,
        fillalpha=0.25, label="0.9 C.I.",
        title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
        yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(node_xs_dd, 1e-3 .* left_pred_stress_il_dd, w=2, label="NN", color="red")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
        label="Measurements", w=2, color="green")
        vline!(support_xs_dd, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)


        t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

        P = plot(t, p2, p1, p4, p3, layout = @layout([A{0.01h}; [grid(2,2)]]), size = (1500, 1000),#=  ylims = (-30,50), =#
            title = ["Test sensor : $(all_sensor_positions[sensor_id]) m " "Left truck lane" "Right truck lane" "" ""], left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    end

    return P
end
plot(dd_stress_il_from_truck.(node_xs_dd,z_truck[2],all_sensor_positions[1],[p_dd]))
function bsciml_prediction_3(sensor_id)

    right_pred_stress_il = stress_il_from_truck_3(p_sciml, all_stress_il[sensor_id],node_xs, meas_xs, all_sensor_positions[sensor_id]; truck_name=RightTruck)
    left_pred_stress_il = stress_il_from_truck_3(p_sciml, all_stress_il[sensor_id],node_xs, meas_xs, all_sensor_positions[sensor_id]; truck_name=LeftTruck)

    right_pred_stress_il_dd = dd_stress_il_from_truck.(node_xs_dd, z_truck[1], all_sensor_positions[sensor_id],[p_dd])
    left_pred_stress_il_dd = dd_stress_il_from_truck.(node_xs_dd, z_truck[2], all_sensor_positions[sensor_id],[p_dd])

    Upper_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bsciml_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bnn_r = [nanquantile(Array(bnn_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(node_xs_dd)]
    Lower_bnn_r = [nanquantile(Array(bnn_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(node_xs_dd)]

    Upper_bnn_l = [nanquantile(Array(bnn_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(node_xs_dd)]
    Lower_bnn_l = [nanquantile(Array(bnn_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(node_xs_dd)]

    begin
        gr()
        p1 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_r, fillrange=1e-3 .* Upper_bsciml_r,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="", yaxis="",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        #= plot(meas_xs, 1e-3 .* MAP_pred_mean_r[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,1,sensor_id], fillrange=1e-3 .* q_u[:,1,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14) =#
        plot!(meas_xs, 1e-3 .* right_pred_stress_il, w=2, label="SciML", color="red")
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)
        #plot!(meas_xs, 0.2 .* all_stress_il[1], ls=:dash, w=2, label="Fe_model")

        p2 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 2, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bsciml_l, fillrange=1e-3 .* Upper_bsciml_l,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        #= plot(meas_xs, 1e-3 .*MAP_pred_mean_l[:,sensor_id], lw=3, label="SciML(MAP)", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* q_l[:,2,sensor_id], fillrange=1e-3 .* q_u[:,2,sensor_id],
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14) =#
        plot!(meas_xs, 1e-3 .* left_pred_stress_il, w=2, label="SciML", color="red")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)

        p3 =
        plot(node_xs_dd, 1e-3 .* bnn_pred_mean[1, :, 1, sensor_id], lw=3, label="BNN", color="blue",
            title="")
        plot!(node_xs_dd, 1e-3 .* Lower_bnn_r, fillrange=1e-3 .* Upper_bnn_r,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(node_xs_dd, 1e-3 .* right_pred_stress_il_dd, w=2, label="NN", color="red")
        plot!(meas_xs, 1e-3 .* all_r_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs_dd, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)


        p4 =
        plot(node_xs_dd, 1e-3 .* bnn_pred_mean[1, :, 2, sensor_id], lw=3, label="BNN", color="blue",
        title="")
        plot!(node_xs_dd, 1e-3 .* Lower_bnn_l, fillrange=1e-3 .* Upper_bnn_l,
        fillalpha=0.25, label="0.9 C.I.",
        title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
        yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(node_xs_dd, 1e-3 .* left_pred_stress_il_dd, w=2, label="NN", color="red")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
        label="Measurements", w=2, color="green")
        vline!(support_xs_dd, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)

        t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

        P = plot(t, p2, p1, p4, p3, layout = @layout([A{0.01h}; [grid(2,2)]]), size = (1500, 1000),#=  ylims = (-30,50), =#
            title = ["Test sensor : $(all_sensor_positions[sensor_id]) m " "Left truck lane" "Right truck lane" "" ""], left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    end

    return P
end

bsciml_prediction_3(7)


right_truck_right_wheel = right_truck.wheel_tvs_pos[1, 1]
right_truck_left_wheel = right_truck.wheel_tvs_pos[1, 2]
left_truck_right_wheel = left_truck.wheel_tvs_pos[1, 1]
left_truck_left_wheel = left_truck.wheel_tvs_pos[1, 2]

xr = collect(meas_xs_less)
zr = collect(range(
    right_truck_right_wheel - 0.1,
    stop=left_truck_left_wheel + 0.1,
    length=38))

bridge_w = 5.700
bridge_L = 295.322

tld = zeros(length(chain_bsciml),length(xr),length(zr))
lateral_load_distr2(x, z, p) = NN2([x ./ bridge_L, z ./ bridge_w], p)[1]
lateral_load_distr3(x, z, s, p) = _NN([x ./ bridge_L, z ./ bridge_w, s ./ last(all_sensor_positions)], p)[1]

iter = 0
for i = 1:1000
    tld[i,:,:] = lateral_load_distr3.(xr,zr',all_sensor_positions[1], [Array(chain_bsciml)[i,:]])
    iter += 1
    show(iter)
end
tld_sciml = lateral_load_distr3.(xr,zr',all_sensor_positions[1],[p_sciml])
tld_scimlmap = lateral_load_distr3.(xr,zr',all_sensor_positions[7],[p_scimlmap])
nanmean_1d(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean_1d, x, dims = y)
tld_mean = nanmean(tld, 1)[1,:,:]
tld_median = median(tld, dims = 1)[1,:,:]
begin
    pp = plot(xr,zr,tld_mean', st=:contourf, fill=true,
        xlabel="Longitudinal direction, x [m]", ylabel="Transverse direction, z [m]",
        title="SciML(MAP): lateral load distribution function",
        #clims = (0,3)
        )

    wheel_zs = [
        right_truck_right_wheel, right_truck_left_wheel,
        left_truck_right_wheel, left_truck_left_wheel
    ]
    hline!(wheel_zs, ls=:dash, c=3, lw=2, label=false)

    vline!(support_xs, color="gray", ls=:dash, label=false)
    vline!([all_sensor_positions[7]], color="red", ls=:dashdot, label=false)
    #annotate!(sp=1, relative(pp[1], sensor_position / support_xs[end], 0.05)...,
        #text("strain gauge", 8, :red, :left, :bottom, rotation=90))
end

#TLD slices
begin
    tld_wheel_slice = zeros(length(chain_bsciml),length(xr),2)
        for i = 1:1000
            tld_wheel_slice[i,:,1] = lateral_load_distr3.(xr,right_truck_center,all_sensor_positions[1], [Array(chain_bsciml)[i,:]])
            tld_wheel_slice[i,:,2] = lateral_load_distr3.(xr,left_truck_center, all_sensor_positions[1], [Array(chain_bsciml)[i,:]])
        end
        tld_slice_up_r = [quantile(tld_wheel_slice[:,i,1],0.95) for i = 1:length(xr)]
        tld_slice_up_l = [quantile(tld_wheel_slice[:,i,2],0.95) for i = 1:length(xr)]
        tld_slice_down_r = [quantile(tld_wheel_slice[:,i,1],0.05) for i = 1:length(xr)]
        tld_slice_down_l = [quantile(tld_wheel_slice[:,i,2],0.05) for i = 1:length(xr)]
        tld_slice = mean(tld_wheel_slice, dims = 1)[1,:,:]

        label = ["right truck center" "left truck center"]
        pp = plot(meas_xs_less,tld_slice,w = 3, label = label)
        plot!(meas_xs_less, tld_slice_down_r, fillrange = tld_slice_up_r,
        fillalpha=0.25, label="0.9 C.I.", color = nothing, fillcolor = "blue"),
        plot!(meas_xs_less, tld_slice_down_l, fillrange = tld_slice_up_l,
        fillalpha=0.25, label="0.9 C.I.", color = nothing, fillcolor = "red")
        vline!([all_sensor_positions[1]], color="green", ls=:dashdot, label=false)
        xlabel!("Longitudinal direction, x [m]")
        ylabel!("mean LLD value")
        title!("BSciML: LLD slices, constant z")
        vline!(support_xs, ls=:dash, label=false, color="black")
        annotate!(sp=1, relative(pp[1], all_sensor_positions[1] / support_xs[end], 0.05)...,
        text("test sensor", 8, :green, :left, :bottom, rotation=90))
    end

begin
        tld_wheel_slices = zeros(38,4)
        tld_wheel_slices[:,1] = lateral_load_distr3.(xr,right_truck_right_wheel,all_sensor_positions[8],[p_sciml])
        tld_wheel_slices[:,2] = lateral_load_distr3.(xr,right_truck_left_wheel,all_sensor_positions[8],[p_sciml])
        tld_wheel_slices[:,3] = lateral_load_distr3.(xr,left_truck_right_wheel,all_sensor_positions[8],[p_sciml])
        tld_wheel_slices[:,4] = lateral_load_distr3.(xr,left_truck_left_wheel,all_sensor_positions[8],[p_sciml])
        label = ["right truck right wheel" "right truck left wheel" "left truck right wheel" "left truck left wheel"]
        plot(meas_xs_less,tld_wheel_slices,w = 3, label = label)
        vline!([all_sensor_positions[9]], color="red", ls=:dashdot, label=false)
        xlabel!("Longitudinal direction, x [m]")
        ylabel!("LLD value")
        title!("MAP: LLD slices, constant z")
end


# TLD std
tld_std = zeros(size(tld[1,:,:]))
for i = 1:length(xr)
    for j = 1:length(zr)

    tld_std[i,j] = sqrt( 1e-3 .* sum(abs2.(tld_mean[i,j] .- tld[:,i,j])))

    end
end


begin
    pp = plot(xr,zr,tld_std, st=:contourf, fill=true,
        xlabel="Longitudinal direction, x [m]", ylabel="Transverse direction, z [m]",
        title="BSciML: LLD standard deviation, H9_S")

    wheel_zs = [
        right_truck_right_wheel, right_truck_left_wheel,
        left_truck_right_wheel, left_truck_left_wheel
    ]
    hline!(wheel_zs, ls=:dash, c=3, lw=2, label=false)

    vline!(support_xs, color="gray", ls=:dash, label=false)
    vline!([all_sensor_positions[1]], color="red", ls=:dashdot, label=false)
    #annotate!(sp=1, relative(pp[1], sensor_position / support_xs[end], 0.05)...,
        #text("strain gauge", 8, :red, :left, :bottom, rotation=90))
end
