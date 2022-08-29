#=

Contains the BSciML model with 3 input BNN.

=#

using LinearAlgebra: \, factorize
using SparseArrays: sparse
using DiffEqFlux: FastChain, FastDense, initial_params, sciml_train
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
using .DataUtils: loadsensordata_sciml, Il_obj, sciml_pred_il_plot, create_prediction_error_plot,
    multi_scale, findall_signchange

meas_path = "/home/edudiro/repo/testproject/RealWorld/Data/measurements_not_processed.csv"
meas_path2 = "/home/edudiro/repo/testproject/RealWorld/Data/measurements_processed.csv"

# ------------------------------------------------------
# SETTINGS & CONTROL
# ------------------------------------------------------

# number of neurons per layer
num_neuron = 20

sensor_position = 65
# steel elastic modulus (should be the same that was used to get stresses from strain measurements)
E = 210e6

# ------------------------------------------------------
# TRUCK LOADING
# ------------------------------------------------------
# Trucks used for loading
right_truck_center = 5.700 / 2 - 3.625 / 2
right_truck = Truck(RightTruck, right_truck_center)

left_truck_center = 5.700 / 2 + 3.625 / 2
left_truck = Truck(LeftTruck, left_truck_center)

# ------------------------------------------------------
# BUILD FEM
# ------------------------------------------------------
# Discretize the girder and get sectional properties
#= max_elem_length = 2.0
girder = girder_sections(
    max_elem_length=max_elem_length * 1e3,
    additional_node_positions=[sensor_position] * 1e3,
)

# Scale to match the units in this file
support_xs = girder["support_xs"] / 1e3
node_xs = girder["node_xs"] / 1e3
elem_c_ys = girder["elem_c_ys"] / 1e3
elem_h_ys = girder["elem_h_ys"] / 1e3
elem_I_zs = girder["elem_I_zs"] / 1e12
elem_EIs = elem_I_zs .* E
elem_W_bottom = elem_I_zs ./ (elem_h_ys .- elem_c_ys)

# Assemble stiffness mx
Ks, nodes, idx_keep = fem_general_single_girder(node_xs, elem_EIs, support_xs)

# convert to a sparse mx and factorize for performance
spar_Ks = sparse(Ks)
fact_spar_Ks = factorize(spar_Ks)

# useful quantities for later
num_dof = size(Ks, 1)
num_node = size(nodes, 1)
sensor_node = findall(sensor_position .== node_xs)[1]

# elastic sectional modulus at sensor location
W_sensor = elem_W_bottom[sensor_node]

# ------------------------------------------------------
# LOAD DATA: TNO MEASUREMENTS (mesurements.csv - not processed)
# ------------------------------------------------------
meas_df = DataFrame(File(meas_path, datarow=3))

# Interpolate the measurements over a shared x grid
# num_meas_x = length(filter(:truck_position => ==("left"), meas_df).x)
num_meas_x = length(node_xs)
meas_xs = LinRange(support_xs[1], support_xs[end], num_meas_x)

truck_positions = levels(meas_df.truck_position)
meas_stress = Array{Float64}(undef, num_meas_x, length(truck_positions))
for (ii, truck_position) in enumerate(truck_positions)
    meas_df_ii = filter(:truck_position => ==(truck_position), meas_df)
    meas_x_ii = meas_df_ii.x
    meas_stress_ii = meas_df_ii.stress * 1e3 # MPa -> kN/m2
    itp = LinearInterpolation(meas_x_ii, meas_stress_ii)
    meas_stress[:, ii] = itp(meas_xs)
end

right_meas_stress_il = meas_stress[:, findfirst("right" .== truck_positions)]
left_meas_stress_il = meas_stress[:, findfirst("left" .== truck_positions)]
meas_xs_less = meas_xs[1:5:end]
node_xs_less = node_xs[1:5:end]
right_meas_stress_il_less = right_meas_stress_il[1:5:end]
left_meas_stress_il_less = left_meas_stress_il[1:5:end]
all_meas_stress_il = vcat(right_meas_stress_il_less,left_meas_stress_il_less)
# ------------------------------------------------------
# COMPUTE STRESS INFL. LINE FOR SENSOR
# ------------------------------------------------------
function stress_at_sensor(sensor_node, load, load_node)
    f = zeros(Float64, num_node * 2)
    f[load_node * 2 - 1] = load
    fs = f[idx_keep]
    us = fact_spar_Ks \ fs
    u = zeros(Float64, num_node * 2)
    u[idx_keep] = us
    m = bending_moments(u, nodes, elem_EIs)
    m_sensor = m[sensor_node]
    stress = m_sensor / W_sensor
    return stress
end

stress_il = stress_at_sensor.(sensor_node, 1.0, 1:num_node)
stress_il_less = stress_il[1:5:end]

# Visualize for checking
begin
    pp = plot(node_xs_less , stress_il_less * 1e3 * 1e-3, # kN -> MN; kN/m2 -> MPa
        xlabel="distance", ylabel="stress [MPa]",
        title="Influence line",
        label="1 MN moving force", legend=:bottomright,
        linewidth=3, linealpha=0.5)
    vline!([sensor_position], linestyle=:dash, label="sensor", c="gray")
end
#savefig(joinpath(IJSSEL_DIR, "plots", "IJssel_stress_il.png")) =#

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
data_sensor_names = ["H1_S", "H3_S", "H5_S", "H8_S"]

data_sensor_positions = [sensor_info[sensor_name] for sensor_name in data_sensor_names]


# Predicted sensor name and position
pred_sensor_names = symdiff(all_sensor_names, data_sensor_names, ["H3_S", "H8_S", "H9_S", "H10_S"]) # Removes "hard" sensors
pred_sensor_positions = [sensor_info[sensor_name] for sensor_name in pred_sensor_names]

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
pred_sensor_nodes = [findall(sensor_position .== node_xs)[1] for sensor_position in pred_sensor_positions]
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
pred_stress_il = [stress_at_sensor_fugro.(sensor_node, f_y, 1:num_node, elem_W_bottom[sensor_node]) for sensor_node in pred_sensor_nodes]
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
pred_stress_il = [pred_stress_il[i][2:end-1] for i in 1:length(pred_stress_il)]
all_stress_il = [all_stress_il[i][2:end-1] for i in 1:length(all_stress_il)]
stress_il_less = [stress_il[i][1:5:end] for i = 1:length(stress_il)]
# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

all_data_r_meas_stress_il = Any[]
all_data_l_meas_stress_il = Any[]

for name in (data_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata_sciml(name, meas_path2, girder)
    push!(all_data_r_meas_stress_il, right_meas_stress_il)
    push!(all_data_l_meas_stress_il, left_meas_stress_il)
end


all_pred_r_meas_stress_il = Any[]
all_pred_l_meas_stress_il = Any[]

for name in (pred_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata_sciml(name, meas_path2, girder)
    push!(all_pred_r_meas_stress_il, right_meas_stress_il)
    push!(all_pred_l_meas_stress_il, left_meas_stress_il)
end

all_r_meas_stress_il = Any[]
all_l_meas_stress_il = Any[]

for name in (all_sensor_names)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata_sciml(name, meas_path2, girder)
    push!(all_r_meas_stress_il, right_meas_stress_il)
    push!(all_l_meas_stress_il, left_meas_stress_il)
end

all_data_r_meas_stress_il_less = [all_data_r_meas_stress_il[i][1:5:end]
                                  for i = 1:length(all_data_r_meas_stress_il)]
all_data_l_meas_stress_il_less = [all_data_l_meas_stress_il[i][1:5:end]
                                  for i = 1:length(all_data_l_meas_stress_il)]

# Store in matrix for the inference phase.
data_matrix = zeros(length(all_data_l_meas_stress_il_less[1]), 2, length(all_data_l_meas_stress_il))

for s = 1:length(all_data_l_meas_stress_il_less)

    data_matrix[:,1,s] = all_data_r_meas_stress_il_less[s]
    data_matrix[:,2,s] = all_data_l_meas_stress_il_less[s]
end

# plot(all_data_r_meas_stress_il_less)
length(all_data_r_meas_stress_il[1])

# Omitted for now
#= function correctsensordata(node_xs, data_sensor_positions, data_sensor_names,
    stress_il, all_data_l_meas_stress_il, all_data_r_meas_stress_il)

    return_l_meas_stress_ils = []
    return_r_meas_stress_ils = []


    for i = 1:length(data_sensor_positions)
        _title = @sprintf("Influence line at %s", data_sensor_names[i])
        corrected_l_meas_stress_il = copy(all_data_l_meas_stress_il[i])
        corrected_r_meas_stress_il = copy(all_data_r_meas_stress_il[i])
        corrected_stress_il = copy(stress_il[i])

        # Get current zero values
        old_l_zerovals_ind = findall_signchange(corrected_l_meas_stress_il, 40, 200) .-1
        if old_l_zerovals_ind[1] != 1
            insert!(old_l_zerovals_ind, 1, 1)
        end

        old_r_zerovals_ind = findall_signchange(corrected_r_meas_stress_il, 40, 200) .-1
        if old_r_zerovals_ind[1] != 1
            insert!(old_r_zerovals_ind, 1, 1)
        end


        # Find new zero values
        new_zerovals_ind = findall(isapprox(0),corrected_stress_il)

        for ind in new_zerovals_ind
            if ind > last(old_l_zerovals_ind)
                    push!(old_l_zerovals_ind, ind)
            end
        end

        for ind in new_zerovals_ind
            if ind > last(old_r_zerovals_ind)
                push!(old_r_zerovals_ind, ind)
            end
        end

        if data_sensor_names[i] == "H3_S"

            insert!(old_r_zerovals_ind, 2, 56)
            insert!(old_l_zerovals_ind, 2, 60)

        end


        new_node_xs_l = multi_scale(collect(node_xs), old_l_zerovals_ind, new_zerovals_ind)
        new_node_xs_r = multi_scale(collect(node_xs), old_r_zerovals_ind, new_zerovals_ind)



        # Update measured data
        itp_l = LinearInterpolation(new_node_xs_l, corrected_l_meas_stress_il)
        itp_r = LinearInterpolation(new_node_xs_r, corrected_r_meas_stress_il)

        new_l_meas_stress_il = collect(itp_l(node_xs))
        new_r_meas_stress_il = collect(itp_r(node_xs))


        # Set analytic zeros
        new_l_meas_stress_il[new_zerovals_ind] .= 0.0
        new_r_meas_stress_il[new_zerovals_ind] .= 0.0

        push!(return_l_meas_stress_ils, new_l_meas_stress_il)
        push!(return_r_meas_stress_ils, new_r_meas_stress_il)


        # Plot updated stress influence lines
        begin
        pp = plot(node_xs , corrected_stress_il * 1e3 * 1e-3, # kN -> MN; kN/m2 -> MPa
        xlabel="distance", ylabel="stress [MPa]",
        title=_title,
        label="1 MN moving force", legend=:bottomright,
        linewidth=lw, linealpha=0.5)
        plot!(pp, node_xs, new_l_meas_stress_il * 1e-3, color = 2, ls = :solid, label = "Left load lane, corrected")
        plot!(pp, node_xs, new_r_meas_stress_il * 1e-3, color = 3, ls = :solid, label = "Right load lane, corrected")
        plot!(pp, corrected_node_xs, corrected_l_meas_stress_il * 1e-3, color = 2, ls = :dot, label = "Left load lane, original")
        plot!(pp, corrected_node_xs, corrected_r_meas_stress_il * 1e-3, color = 3, ls = :dot, label = "Right load lane, original")
        vline!([data_sensor_positions[i]], linestyle=:dash, label="sensor", c="gray")
        end
        display(pp)
        savename = @sprintf("IJssel_stress_il%s.png", data_sensor_names[i])
        savefig(joinpath(IJSSEL_DIR, "plots", savename))
    end
    return (return_l_meas_stress_ils, return_r_meas_stress_ils)
end =#


# Correct the measured sensor data
#= (all_data_l_meas_stress_il, all_data_r_meas_stress_il) = correctsensordata(node_xs, data_sensor_positions, data_sensor_names,
                                stress_il, all_data_l_meas_stress_il, all_data_r_meas_stress_il)


(all_pred_l_meas_stress_il, all_pred_r_meas_stress_il) = correctsensordata(node_xs, pred_sensor_positions, pred_sensor_names,
                                pred_stress_il, all_pred_l_meas_stress_il, all_pred_r_meas_stress_il) =#




# ------------------------------------------------------
# DEFINE NEURAL NETWORK
# ------------------------------------------------------
# The neural network NN(x,z) is defined and the weights initialized.
NN = FastChain(
    FastDense(3, 64, swish),
    FastDense(64, 64, swish),
    FastDense(64, 32, swish),
    FastDense(32, 1),
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

function stress_il_from_truck(p, stress_il, il_xs, sensor_pos; truck_name::TruckName)
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

    lateral_load_distr(x, z, s) = NN([x ./ bridge_L, z ./ bridge_w, s ./ last_sensor], p)[1]

    force_il_r = lateral_load_distr.(node_xs, right_wheel_z, sensor_pos)
    force_il_l = lateral_load_distr.(node_xs, left_wheel_z, sensor_pos)

    stress_il_r = stress_il .* force_il_r
    stress_il_l = stress_il .* force_il_l

    interp_stress_il_r = linterp_even(node_xs, stress_il_r, 0.0)
    interp_stress_il_l = linterp_even(node_xs, stress_il_l, 0.0)

    function stress_from_truck(x)
        truck_stress = sum(interp_stress_il_r.(right_wheel_x .+ x) .* right_wheel_forces) +
                       sum(interp_stress_il_l.(left_wheel_x .+ x) .* left_wheel_forces)
        return truck_stress
    end
    pred_stress_il = stress_from_truck.(il_xs)

    return pred_stress_il
end

function sciml_get_mse(p, _truck_name, sensor_positions, meas_xs)
    mse = 0.0
    len = length(sensor_positions)

    if _truck_name == RightTruck
        for i = 1:len
            pred_il = stress_il_from_truck(p, stress_il_less[i], meas_xs_less, sensor_positions[i]; truck_name=_truck_name)
            mse += mean(abs2.(pred_il .- (all_data_r_meas_stress_il_less[i])))

        end
    elseif _truck_name == LeftTruck
        for i = 1:len
            pred_il = stress_il_from_truck(p, stress_il_less[i], meas_xs_less, sensor_positions[i]; truck_name=_truck_name)
            mse += mean(abs2.(pred_il .- (all_data_l_meas_stress_il_less[i])))
        end
    end

    return mse / len
end

function loss(p)
    mse_right = sciml_get_mse(p, RightTruck, data_sensor_positions, meas_xs_less) * 1e-6 # MPa^2
    mse_left = sciml_get_mse(p, LeftTruck, data_sensor_positions, meas_xs_less) * 1e-6 # MPa^2
    mse = mse_right + mse_left
    # mse = mse_left
    # mse = mse_right
    return mse
end

# ------------------------------------------------------
# FIT NEURAL NETWORK
# ------------------------------------------------------
max_iters = 200
res = sciml_train(loss, p_init,
    ADAM(0.05), maxiters=max_iters, save_best=true, cb=callback)

# refine with a lower learning rate
res = sciml_train(loss, res.minimizer,
    ADAM(0.003), maxiters=max_iters, save_best=true, cb=callback)

p = res.minimizer

using JLD
save("RealWorld/Results/First_versions/sciml_4sensor_3input_38points.jls", "p", p)

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
# for the 4 load lanes (2 truck lanes and 2 wheel lanes)
# They are all very similar though.
# Also generate with different number of obs.
# The number of elements are the number of data sensors to be stored.
right_wheel_il_less = Array{Vector}(undef, length(stress_il_less))
left_wheel_il_less = Array{Vector}(undef, length(stress_il_less))
right_wheel_il = Array{Vector}(undef, length(stress_il))
left_wheel_il = Array{Vector}(undef, length(stress_il))

for i = 1:length(stress_il_less) #loop over data sensors

    (right_wheel_right_truck_less, left_wheel_right_truck_less) = stress_il_from_truck_bnn(meas_xs_less, node_xs_less, stress_il_less[i]; truck_name=RightTruck)
    (right_wheel_right_truck, left_wheel_right_truck) = stress_il_from_truck_bnn(meas_xs, node_xs, stress_il[i]; truck_name=RightTruck)
    (right_wheel_left_truck_less, left_wheel_left_truck_less) = stress_il_from_truck_bnn(meas_xs_less, node_xs_less, stress_il_less[i]; truck_name=LeftTruck)
    (right_wheel_left_truck, left_wheel_left_truck) = stress_il_from_truck_bnn(meas_xs, node_xs, stress_il[i]; truck_name=LeftTruck)

    #Pass to matrix
    right_wheel_il_less[i] = [right_wheel_right_truck_less, right_wheel_left_truck_less]
    left_wheel_il_less[i] = [left_wheel_right_truck_less, left_wheel_left_truck_less]
    right_wheel_il[i] = [right_wheel_right_truck, right_wheel_left_truck]
    left_wheel_il[i] = [left_wheel_right_truck, left_wheel_left_truck]

end
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

num_param = length(initial_params(NN))
alpha = 0.09
sig = sqrt(1.0 / alpha)

@model function bayesian_sciml_realworld(observations, x, Right_wheel_il, Left_wheel_il,
    ::Type{T}=Float64) where {T}

    bridge_w = 5.700
    bridge_L = 295.322

    #prior parameters
    θ ~ MvNormal(zeros(num_param), sig .* ones(num_param))
    std ~ InverseGamma(3, 4)

    #TLD definition and constants
    lateral_load_distr(x, z, s) = NN([x ./ bridge_L, z ./ bridge_w, s ./ last(all_sensor_positions)], θ)[1]


    pred_il = Array{T}(undef, length(x), 2, length(Right_wheel_il))

    #Necessary to dintinguish between inference
    #and prediction due to the different number
    #of sensors used in both cases.

    if ismissing(observations) == false #if inference
        for i = 1:length(Right_wheel_il) #loops over sensors
            #Compute TLD and Fe_model * TLD

            TLD_r = lateral_load_distr.(x, right_truck_center, data_sensor_positions[i])
            TLD_l = lateral_load_distr.(x, left_truck_center, data_sensor_positions[i])

            pred_il[:, 1, i] = TLD_r .* (Right_wheel_il[i][1] + Left_wheel_il[i][1])
            pred_il[:, 2, i] = TLD_l .* (Right_wheel_il[i][2] + Left_wheel_il[i][2])

        end

    else # if predicting
        for i = 1:length(Right_wheel_il) #loops over sensors
            #Compute TLD and Fe_model * TLD

            TLD_r = lateral_load_distr.(x, right_truck_center, all_sensor_positions[i])
            TLD_l = lateral_load_distr.(x, left_truck_center, all_sensor_positions[i])

            pred_il[:, 1, i] = TLD_r .* (Right_wheel_il[i][1] + Left_wheel_il[i][1])
            pred_il[:, 2, i] = TLD_l .* (Right_wheel_il[i][2] + Left_wheel_il[i][2])

        end
    end

    #Likelihood
    observations ~ MvNormal(pred_il[:], 100 * std)

end

infer_model_bsciml_realworld = bayesian_sciml_realworld(data_matrix[:], meas_xs_less, right_wheel_il_less, left_wheel_il_less);

#Check model works
mle_model = optimize(infer_model_bsciml_realworld, MLE())
mle_params = mle_model.values.array

lateral_load_distr(x, z, s) = NN([x ./ bridge_L, z ./ bridge_w, s ./ last(all_sensor_positions)], mle_params)[1]
    bridge_w = 5.700
    bridge_L = 295.322
    right_wheel_z = [right_truck.wheel_tvs_pos[1, 1],left_truck.wheel_tvs_pos[1,1]]
    left_wheel_z = [right_truck.wheel_tvs_pos[1, 2],left_truck.wheel_tvs_pos[1,2]]
    pred_il = zeros(length(meas_xs_less),2,1)
    for i=1:1
        #Compute TLD and Fe_model * TLD
        for j = 1:2
            TLD_r = lateral_load_distr.(meas_xs_less , right_wheel_z[j], data_sensor_positions[1])
            TLD_l = lateral_load_distr.(meas_xs_less , left_wheel_z[j], data_sensor_positions[1])
            pred_il[:,j,i] = (right_wheel_il_less[i][j] .* TLD_r) + (right_wheel_il_less[i][j] .* TLD_l)
        end
    end

pred_il[:,1,1]
plot(meas_xs_less, pred_il[:,1,1])
plot!(meas_xs_less, all_data_r_meas_stress_il_less[1])

#Inference
chain_bsciml = sample(infer_model_bsciml_realworld, NUTS(0.65), 1000, init_params = [mle_params])
#write("RealWorld/Results/First_versions/init_params_4sensor_chain_bsciml_3input_38points.jls", chain_bsciml)

# Prediction model
chain_bsciml = read("RealWorld/Results/First_versions/init_params_4sensor_chain_bsciml_3input_38points.jls", Chains)
test_model_bsciml_realworld = bayesian_sciml_realworld(missing, meas_xs, right_wheel_il_all, left_wheel_il_all);

# ------------------------------------------------------
# VISUALIZE
# ------------------------------------------------------

#= plot(left_pred_stress_il)
plot!(all_data_l_meas_stress_il[1]) =#

bsciml_pred = reshape(Array(predict(test_model_bsciml_realworld, chain_bsciml)), (1000, length(meas_xs), 2, 9))
bsciml_pred_mean = mean(bsciml_pred, dims=1)
bsciml_median = median(bsciml_pred, dims = 1)
plot(meas_xs, bsciml_pred_mean[1,:,1,1])
plot!(meas_xs, stress_il_from_truck(p, stress_il[1], meas_xs, all_sensor_positions[1]; truck_name=RightTruck))
plot!(meas_xs_less, all_data_r_meas_stress_il_less[1])


gr()
group(chain_bsciml, :std)
plot(chain_bsciml, :std)
plot(chain_bsciml[:,500:501,:])

# ......................................................
# Measured and predicted influence lines
# ......................................................
#= y_scale = 1e-3 # kN/m2 -> MPa
begin
    lw = 3
    pp = plot(title="Influence line",
        xlabel="Position of the first axle [m]", ylabel="Stress [MPa]",
        legend=:topright)

    # Right truck loading
    plot!(meas_xs, right_meas_stress_il * y_scale,
        linewidth=lw, color=1, linealpha=0.5, label="right, measured")
    plot!(meas_xs, right_pred_stress_il * y_scale,
        linewidth=lw, color=1, label="right, SciML", ls=:dash)

    # Left truck loading
    plot!(meas_xs, left_meas_stress_il * y_scale,
        linewidth=lw, color=2, linealpha=0.5, label="left, measured")
    plot!(meas_xs, left_pred_stress_il * y_scale,
        linewidth=lw, color=2, label="left, SciML", ls=:dash)

    vline!([sensor_position], color="red", ls=:dashdot, label=false)
    annotate!(sp=1, relative(pp[1], sensor_position / support_xs[end], 0.05)...,
        text("strain gauge", 8, :red, :left, :bottom, rotation=90))
        vline!(support_xs, color="gray", ls=:dash, label=false)
end =#


nanquantile(x, q) = quantile(filter(!isnan, x), q)

function bsciml_prediction(sensor_id)

    right_pred_stress_il = stress_il_from_truck(p, all_stress_il[sensor_id], meas_xs, all_sensor_positions[sensor_id]; truck_name=RightTruck)
    left_pred_stress_il = stress_il_from_truck(p, all_stress_il[sensor_id], meas_xs, all_sensor_positions[sensor_id]; truck_name=LeftTruck)

    Upper_bnn_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bnn_r = [nanquantile(Array(bsciml_pred)[:, i, 1, sensor_id], 0.05) for i = 1:length(meas_xs)]

    Upper_bnn_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.95) for i = 1:length(meas_xs)]
    Lower_bnn_l = [nanquantile(Array(bsciml_pred)[:, i, 2, sensor_id], 0.05) for i = 1:length(meas_xs)]

    begin
        gr()
        p1 =
        plot(meas_xs, 1e-3 .* bsciml_pred_mean[1, :, 1, sensor_id], lw=3, label="BSciML", color="blue",
            title="")
        plot!(meas_xs, 1e-3 .* Lower_bnn_r, fillrange=1e-3 .* Upper_bnn_r,
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
        plot!(meas_xs, 1e-3 .* Lower_bnn_l, fillrange=1e-3 .* Upper_bnn_l,
            fillalpha=0.25, label="0.9 C.I.",
            title="", xlabel="Longitudinal direction, x [m]", yaxis="Stress [MPa]",
            yflip=false, color=nothing, fillcolor="blue", xguidefontsize=14, yguidefontsize=14)
        plot!(meas_xs, 1e-3 .* left_pred_stress_il, w=2, label="SciML", color="red")
        plot!(meas_xs, 1e-3 .* all_l_meas_stress_il[sensor_id],
            label="Measurements", w=2, color="green")
        vline!(support_xs, ls=:dash, label=false, color="black")
        hline!([0], label=false, color="black")
        scatter!([all_sensor_positions[sensor_id]], [0], label=false)

        t = plot( title = " ", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

        P = plot(t, p2, p1, layout = @layout([A{0.01h}; [B C]]), size = (1500, 500),#=  ylims = (-30,50), =#
            title = ["Test sensor : $(all_sensor_positions[sensor_id]) m " "Left truck lane" "Right truck lane"], left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    end

    return P
end

bsciml_prediction(8)

# FROM THIS POINT ON, NO WORK HAS BEEN DONE YET FOR THE BSCIML PRED.

# Lateral load distribution function
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
lateral_load_distr(x, z, s, p) = NN([x ./ bridge_L, z ./ bridge_w, s ./ last(all_sensor_positions)], p)[1]
iter = 0
for i = 1:1000
    tld[i,:,:] = lateral_load_distr.(xr,zr',all_sensor_positions[8], [Array(chain_bsciml)[i,:]])
    iter += 1
    show(iter)
end
tld_sciml = lateral_load_distr.(xr,zr',all_sensor_positions[1],[p])
nanmean_1d(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean_1d, x, dims = y)
tld_mean = nanmean(tld, 1)[1,:,:]
tld_median = median(tld, dims = 1)[1,:,:]
begin
    pp = plot(xr,zr,tld_mean, st=:contourf, fill=true,
        xlabel="Longitudinal direction, x [m]", ylabel="Transverse direction, z [m]",
        title="BSciML: lateral load distribution function, s = H9_S")

    wheel_zs = [
        right_truck_right_wheel, right_truck_left_wheel,
        left_truck_right_wheel, left_truck_left_wheel
    ]
    hline!(wheel_zs, ls=:dash, c=3, lw=2, label=false)

    vline!(support_xs, color="gray", ls=:dash, label=false)
    vline!([all_sensor_positions[8]], color="red", ls=:dashdot, label=false)
    #annotate!(sp=1, relative(pp[1], sensor_position / support_xs[end], 0.05)...,
        #text("strain gauge", 8, :red, :left, :bottom, rotation=90))
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
        title="BSciML: LLD standard deviation, s = H9_S")

    wheel_zs = [
        right_truck_right_wheel, right_truck_left_wheel,
        left_truck_right_wheel, left_truck_left_wheel
    ]
    hline!(wheel_zs, ls=:dash, c=3, lw=2, label=false)

    vline!(support_xs, color="gray", ls=:dash, label=false)
    vline!([all_sensor_positions[8]], color="red", ls=:dashdot, label=false)
    #annotate!(sp=1, relative(pp[1], sensor_position / support_xs[end], 0.05)...,
        #text("strain gauge", 8, :red, :left, :bottom, rotation=90))
end
