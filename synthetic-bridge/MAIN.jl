#=
Add physical knowledge of the system to the BNN.
The BNN will predict the transverse load
distribution function, and we will then find the load
on the girder by introducing the influence lines modelled
by the FE method.

The ground truth now is the predicted transverse load
dist. function.
The training data are the synthetic influence lines on
the girder.
The FE solver finds the influence lines at any position
in the bridge x direction.
Inside data_generator we include all these calculations.
Therefore, the output of this function regarding the
influence lines at any x must be called inside
turing @model.
=#

using AbstractMCMC
using Distributed
using DiffEqFlux
using Flux
using Zygote
using Optim
using LinearAlgebra
using SparseArrays
using Statistics
using StatsPlots
using Test
using Printf
using BenchmarkTools
using Random
using Distributions
using Turing
using Interpolations
using JLD
include("CALFEMhelpers.jl")
include("DevelopInterval.jl")
include("TrainingFunctionsDataDriven.jl")
include("TrainingFunctionsSciML.jl")
include("BNN.jl")
include("InferenceFunctions.jl")


##Build bridge model
#= return data matrix containing
influence line from a set of load lanes
at any chosen sensor or set of sensors.
=#

Turing.setadbackend(:tracker)
Random.seed!(1234)
# Problem set-up

# Define sensor locations
L = 60.0
steps = 2.5 # size of steps for FEM
sensor_locs = collect(0:steps:L)
x_middle = L / 2
# Define locations of load lanes
load_lane_locs = collect(range(0,5,length = 5))

# Dimensions for plots and comparisons NB. that length(sensor_locs) == number of segments i FE model
xdim = length(sensor_locs)
zdim = 20
zr = range(0, stop = 5, length = zdim)

data_train, noisydata_train, inf_l, ground_truth =
data_generator([load_lane_locs[1]],[sensor_locs[13]], zdim, steps);
plot(noisydata_train[:,1,1])



#####################################################
# Training for nn, sciml,bnn,bnn-sciml
#####################################################

#NN training

nn_params = trainsystemDD([load_lane_locs[1]], [sensor_locs[13]], noisydata_train, zdim)
#save("Results/Case 8.2/nn_params.jld", "nn_params", nn_params)

#SCIML NN training
sciml_params = trainsystemSciML([load_lane_locs[1]], [sensor_locs[13]], noisydata_train, zdim)
#save("Results/Case 8.2/sciml_params.jld", "sciml_params", sciml_params)


#INFERENCE
num_samples = 1000

#Call models
infer_model_sciml = bayesian_sciml(noisydata_train[:], sensor_locs, [load_lane_locs[1]],[sensor_locs[13]], num_params, inf_l);

infer_model_bnn = bayesian_dd(100 .* noisydata_train[:],sensor_locs,[load_lane_locs[1]],[sensor_locs[13]]);

#MLE for sciml
begin

    mle_model = optimize(infer_model_sciml, MLE())
    mle_params = mle_model.values.array

    plot(inf_l[:,1].*bnn.(sensor_locs,load_lane_locs[1], x_middle, [mle_params]))
    plot!(noisydata_train[:,1,1])

    #= plot(bnn.(sensor_locs,load_lane_locs[1], x_middle, [mle_params]))
    plot!(ground_truth[:,1,1]) =#

end


#Perform sampling.
chain_sciml = sample(infer_model_sciml, NUTS(.65), num_samples, init_params = mle_params)
chain_bnn = sample(infer_model_bnn, NUTS(.65), num_samples)

#Save
#write("Results/Accuracy_comparison/18_sensors/chain_sciml.jls", chain_sciml)
#write("Results/Accuracy_comparison/2_sensors/chain_bnn.jls", chain_bnn)
#print("saved chains")

## PREDICTIONS

#= #Quick prediction check
#If more than one load lane was chose, then plots
#the flatten prediction
begin
    plot(noisydata_train[:,:,1][:], label = "training data")
    plot!(pred_mean, label = "mean_pred",
    xticks = ([0,13,26,39,52,65]),
    xlabel = "#obs",
    ylabel = "deflection[mm]")
    vline!([0,13,26,39,52,65], label = "")

end =#

##Quantify goodness of fit
#compute all predictions loss
#= loss = [sum(abs2.(noisydata_train[:,1,1] - Array(pred_bnn)[i,:])) for i = 1:size(Array(pred_sciml))[1]]
mean(loss)

import StatsPlots
StatsPlots.histogram(loss) =#


######################################################
####################PLOTTING##########################
######################################################


#= begin
    plotlyjs() #set backend
    plot(sensor_locs, pred_mean, w = 5,
        title = "(30,0)")
    for k = 1:10:num_samples
        plot!(sensor_locs, Array(pred_sciml)[k, :], alpha = 1,
            color = "#BBBBBB", legend = false)
    end
    plot!(sensor_locs, noisydata_train[:,1,1], w = 4, color = :red,
        xlabel = "x[m]", ylabel = "deflection [mm]")
    plot!(sensor_locs, pred_mean, w = 4)
    vline!([0.0, 20.0, 40.0, 60.0])
end =#
#check the infered obs. error and θ convergence
gr()

plot(chain_sciml[:, 54:57, :])
plot(Array(chain["acceptance_rate"]))
scatter!(chain["is_accept"])
plot(chain["step_size"])
chain["is_accept"]
chain["step_size"]

# parameter standard deviation
error = std(Array(chain), dims = 1)[:]
histogram(error, bins = 20, xlabel = "Parameter Standard Deviation",
legend = false)

#Average absolute deviations.
begin
    aad = zeros(size(Array(chain))[2])

    for i = 1:size(Array(chain))[2]

        a = sum(abs.(Array(chain)[:,i] .- mean(Array(chain),dims = 1)[i]))
        aad[i] = (a/size(Array(chain))[1])

    end
end
histogram(aad, bins = 20, xlabel = "Parameter average absolute deviation",
legend = false)

group(chain_sciml, :std)
plot(chain_sciml, :std)
