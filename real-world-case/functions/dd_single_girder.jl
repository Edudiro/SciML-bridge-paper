#=
Overview

Application of data-driven ML to predict stress influence lines of the IJssel bridge.

The objective is to find the influence surface prediction function at for sensor a placed along the bridge x-axis,
     using a neural network and measurements.

Textual description

* a single girder is considered
* the LLD is function fitted using indirect measurements (stresses)
* the LLD function is modelled with a neural network (NN)


Using a neural network, the complex behavior of the real world bridge is captured.

Math description

LLD function:

`stress_pred(x,z,a) = NN(x,z,a,p)`

where `p` is the vector of parameters of the NN. From here on, for brevity we indicate
only the dependence on `p`: `NN(p)`.

We aim to minimize the discrepancy between measurements and predictions, this is an
optimization problem:

`p_opt = argmin(sum(stress_pred(x) - stress_meas(x))^2)`

Coordinates

- `x`   longitudinal axis of the girder
- `a`   sensor point along the longitudinal axis of the girder
- `z`   transverse axis of the girder/bridge
- `y`   depth (vertical) axis of the girder

Units

Unless otherwise stated:
- length: [m]
- force: [kN]

TODO
- add convergence plots =#
using LinearAlgebra: \, factorize
using SparseArrays: sparse
using DiffEqFlux: FastChain, FastDense, initial_params, sciml_train
using Flux: swish, tanh, ADAM
using Statistics: mean
using Interpolations: LinearInterpolation
using CSV: File
using DataFrames: DataFrame, filter, levels
using StatsPlots
using Printf: @sprintf
using Turing
using Optim
Turing.setadbackend(:tracker)
#= include(abspath(joinpath("continuous_girder", "analysis", "Configs.jl")))
using .Configs: ANALYSIS_DIR, FEM_DIR, IJSSEL_DIR =#

include( "Utils.jl")
using .Utils: linterp_even, relative

include("FEMGirders.jl")
using .FEMGirders: fem_general_single_girder, bending_moments

include( "GirderSections.jl")
using .GirderSections: girder_sections

include("FieldTest.jl")
using .FieldTest: Truck, TruckName, RightTruck, LeftTruck

include("DataPlotting.jl")
using .DataUtils: loadsensordata, Il_obj, dd_pred_il_plot

meas_path = "/home/edudiro/repo/testproject/RealWorld/Data/measurements_processed.csv"

# ------------------------------------------------------
# SETTINGS & CONTROL
# ------------------------------------------------------
max_elem_length = 2.0

# number of neurons per layer
num_neuron = 20
#32 neurons for  deterministic nn

# steel elastic modulus (should be the same that was used to get stresses from strain measurements)
E = 210e6

# Store sensor information
sensor_info = Dict("H1_S" =>20.42, "H2_S" =>34.82, "H3_S" =>47.700, "H4_S" =>61.970,
                    "H5_S" =>68.600, "H7_S" =>96.800, "H8_S" =>113.9, "H9_S" =>123.900, "H10_S" =>147.500)


# Data sensor name and position
data_sensor_names = ["H1_S"]#, "H3_S", "H5_S", "H8_S"]#, "H5_S", "H8_S", "H9_S", "H10_S"]
data_sensor_positions = [sensor_info[sensor_name] for sensor_name in data_sensor_names]

# Predicted sensor name and position
pred_sensor_name =  "H7_S"
pred_sensor_position = sensor_info[pred_sensor_name]

# ------------------------------------------------------
# TRUCK LOADING
# ------------------------------------------------------
# Trucks used for loading
right_truck_center = 5.700 / 2 - 3.625 / 2
right_truck = Truck(RightTruck, right_truck_center)

left_truck_center = 5.700 / 2 + 3.625 / 2
left_truck = Truck(LeftTruck, left_truck_center)

z_truck = [right_truck_center,left_truck_center]

# ------------------------------------------------------
# BUILD FEM
# ------------------------------------------------------
# Discretize the girder and get sectional properties
girder = girder_sections(
    max_elem_length=max_elem_length * 1e3,
    additional_node_positions=data_sensor_positions * 1e3,
)

# Scale to match the units in this file
support_xs = girder["support_xs"] / 1e3
node_xs = girder["node_xs"] / 1e3

# Interpolate the measurements over a shared x grid
num_meas_x = length(node_xs)
meas_xs = LinRange(support_xs[1], support_xs[end], num_meas_x)



# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

#sensor_array = Any[]
all_r_meas_stress_il = Any[]
all_l_meas_stress_il = Any[]

for name in (data_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata(name, meas_path, girder)
    push!(all_r_meas_stress_il, right_meas_stress_il)
    push!(all_l_meas_stress_il, left_meas_stress_il)
end

meas_stress_il_matrix = zeros(length(node_xs),2,length(all_l_meas_stress_il))
for i = 1:length(all_r_meas_stress_il)
    meas_stress_il_matrix[:,1,i] = all_r_meas_stress_il[i]
    meas_stress_il_matrix[:,2,i] = all_l_meas_stress_il[i]
end
maximum(meas_stress_il_matrix[:,2,:])

#Choose less measurements.
#interpolate
#= itp_x = collect(range(support_xs[1],support_xs[end],length = 25))
itp_meas = zeros(length(itp_x), length(z_truck), length(all_l_meas_stress_il))
    for i = 1:length(z_truck)
        for j = 1:length(all_l_meas_stress_il)
            itp = LinearInterpolation(meas_xs,meas_stress_il_matrix[:, i, j])
            itp_meas[:, i, j] = itp.(itp_x)
        end
    end
 =#

meas_stress_il_matrix_less = meas_stress_il_matrix[1:5:end,:,:]
node_xs_less = node_xs[1:5:end]
meas_stress_il_matrix
plot(node_xs_less,meas_stress_il_matrix_less[:,1,1])
plot!(node_xs,meas_stress_il_matrix[:,1,1])

# ------------------------------------------------------
# DEFINE NEURAL NETWORK
# ------------------------------------------------------
# The neural network NN(x,z) is defined and the weights initialized.
NN= FastChain(
                FastDense(3,num_neuron,swish),
                FastDense(num_neuron,num_neuron,swish),
                FastDense(num_neuron,1),
)

p_init = initial_params(NN)

# ------------------------------------------------------
# DEFINE LOSS FUNCTION
# ------------------------------------------------------
# Define callback to help visualizing the results
function callback(p, l)
    # Turn off for optimizing
    @show l
    false
end

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
    mse = dd_get_mse(node_xs, z_truck, data_sensor_positions,[p]) * 1e-6 # MPa^2
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

p = res.minimizer


# ------------------------------------------------------
# PURE BNN
# ------------------------------------------------------
num_param = length(initial_params(NN))
alpha = 0.09
sig = sqrt(1.0 / alpha)

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


#= test_y = zeros(length(itp_x), length(z_truck), length(all_l_meas_stress_il))
test_y[:,1,1] = sin.(itp_x)
test_y[:,2,1] = cos.(itp_x) =#

infer_model_bnn_realworld = bayesian_dd_realworld(meas_stress_il_matrix_less[:], node_xs_less, z_truck,data_sensor_positions);
mle_model = optimize(infer_model_bnn_realworld, MLE())
mle_params = mle_model.values.array

mle_pred = plot(node_xs_less, dd_stress_il_from_truck.(node_xs_less, z_truck[1], data_sensor_positions[1],[mle_params]))
plot!(node_xs, meas_stress_il_matrix[:,2,1])
#= append!(res.minimizer,0.05*5000) =#

chain_bnn = sample(infer_model_bnn_realworld, NUTS(.65), 1000, init_params = [mle_params])

write("RealWorld/Results/Document/H1-bnn-33points-500p.jls", chain_bnn)
#chain_bnn = read("RealWorld/Results/First_versions/4sensors_chain_bnn_33_nodes.jls", Chains)

test_model_bnn_realworld = bayesian_dd_realworld(missing, node_xs_less, z_truck, data_sensor_positions);


# ------------------------------------------------------
# VISUALIZE AT ONE OF THE DATA SENSORS
# ------------------------------------------------------

# deterministic dd
dd_pred_stress_il = zeros(length(node_xs),2,length(data_sensor_positions))
for i = 1:length(data_sensor_positions)
    for j = 1:2
        dd_pred_stress_il[:,j,i] = dd_stress_il_from_truck.(node_xs, z_truck[j], data_sensor_positions[i],[p])
    end
end

plot(node_xs,dd_pred_stress_il[:,1,1])
plot!(node_xs,meas_stress_il_matrix[:,1,1])

plot!(node_xs,dd_pred_stress_il[:,2,1])
plot!(node_xs,meas_stress_il_matrix[:,2,1])

#Bayesian dd

nanmean_1d(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean_1d, x, dims = y)

bnn_pred = reshape(Array(predict(test_model_bnn_realworld,chain_bnn)),(1000,length(node_xs_less),2,1))
bnn_pred_mean = mean(bnn_pred, dims = 1)
plot(node_xs_less,bnn_pred_mean[1,:,1,1])
plot!(node_xs,meas_stress_il_matrix[:,1,1])
plot!(itp_x, bnn_pred[500,:,1])
gr()
group(chain_bnn, :std)
plot(chain_bnn, :std)
plot(chain_bnn[:,50,:])


nanquantile(x, q) = quantile(filter(!isnan, x), q)

Upper_bnn = [nanquantile(Array(bnn_pred)[:, i,2,1], 0.95) for i = 1:length(node_xs)]
Lower_bnn = [nanquantile(Array(bnn_pred)[:, i,2,1], 0.05) for i = 1:length(node_xs)]
begin
    plot(node_xs, 1e-3 .* bnn_pred_mean[1,:,2,1], lw = 3, label = "BNN", color = "blue")
    plot!(node_xs, 1e-3 .* Lower_bnn, fillrange = 1e-3 .* Upper_bnn,
        fillalpha = 0.25, label = "0.9 C.I.",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "Stress [MPa]",
        yflip = false, color = nothing, fillcolor = "blue", xguidefontsize = 14, yguidefontsize = 14)
    plot!(node_xs, 1e-3 .* dd_pred_stress_il[:,2,1], w = 2, label = "NN", color = "red")
    plot!(node_xs,1e-3 .* meas_stress_il_matrix[:,2,1],
    label = "Measurements", ls =:dashdot, w = 2, color = "green")
    vline!(support_xs, ls =:dash, label = false, color = "black")
    hline!([0], label = false, color = "black")
    scatter!([data_sensor_positions[1]],[0], label = false)

end


#= # ......................................................
# Measured and predicted influence lines at data sensor
# ......................................................
y_scale = 1e-3 # kN/m2 -> MPa
all_stress = vcat(P1_right_meas_stress_il, P1_left_meas_stress_il)
lims = [minimum(all_stress), maximum(all_stress)] .* y_scale
lims[1] -=18
lims[2] +=4
il = Il_obj(meas_xs, support_xs, P1_right_meas_stress_il, P1_right_pred_stress_il, P1_left_meas_stress_il, P1_left_pred_stress_il);
display(dd_pred_il_plot(il, P1_pred_sensor_name, P1_pred_sensor_position, data_sensor_positions, lims))
#savefig(abspath(joinpath(IJSSEL_DIR, "plots", "IJssel_DD_single_girder_stress_il_at_datasensor.png")))





# ------------------------------------------------------
# VISUALIZE AT PREDICTED SENSOR
# ------------------------------------------------------
P2_pred_sensor_name = "H7_S"
P2_pred_sensor_position = sensor_info[P2_pred_sensor_name]
(P2_right_meas_stress_il, P2_left_meas_stress_il) = loadsensordata(P2_pred_sensor_name, meas_path, girder)
P2_right_pred_stress_il = dd_stress_il_from_truck(p, "right", P2_pred_sensor_position)
P2_left_pred_stress_il = dd_stress_il_from_truck(p, "left", P2_pred_sensor_position)

il = Il_obj(meas_xs, support_xs, P2_right_meas_stress_il, P2_right_pred_stress_il, P2_left_meas_stress_il, P2_left_pred_stress_il);
display(dd_pred_il_plot(il, P2_pred_sensor_name, P2_pred_sensor_position, data_sensor_positions, lims))
savefig(abspath(joinpath(IJSSEL_DIR, "plots", "IJssel_DD_single_girder_stress_il_at_predsensor.png"))) =#
