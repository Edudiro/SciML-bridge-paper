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
include("VisualizeFuncs.jl")
include("BNN.jl")
include("InferenceFunctions.jl")

Random.seed!(1234)
# Define sensor locations
L = 60.0
steps = 1.25 # size of steps for FEM
sensor_locs = collect(0:steps:L)
x_middle = L / 2
# Define locations of load lanes
load_lane_locs = collect(range(0, 5, length = 5))

# Dimensions for plots and comparisons NB. that length(sensor_locs) == number of segments i FE model
xdim = length(sensor_locs)
zdim = 20
zr = range(0, stop = 5, length = zdim)

## Reconstruct results
# Retrieve all synthetic data for result comparison.
data_test, noisydata_test, inf_l_test, ground_truths = data_generator(load_lane_locs, sensor_locs, zdim, steps);



#load chain
chain_sciml = read("Results/Accuracy_comparison/21_sensors/chain_sciml.jls", Chains)
chain_bnn = read("Results/Accuracy_comparison/25_sensors/chain_bnn.jls", Chains)
group(chain_sciml, :std)
#predict model
test_model_sciml = bayesian_sciml(missing, sensor_locs, load_lane_locs[1],
    sensor_locs, num_params, inf_l_test);
test_model_bnn = bayesian_dd(missing, sensor_locs, load_lane_locs[1],
    sensor_locs);

    #if nan values use the following:
nanmean_1d(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean_1d, x, dims = y)

#bnn
pred_bnn = 1000 .* Array(predict(test_model_bnn, chain_bnn)) ./ 100
pred_bnn = reshape(pred_bnn, (1000, 49, 49))
pred_mean_bnn = [vec(nanmean(pred_bnn[:, :, i], 1)) for i = 1:length(sensor_locs)]
plot(pred_mean_bnn[25])

#bnn-sciml
pred_scimlbnn = reshape(1000 .* Array(predict(test_model_sciml, chain_sciml)), (1000, 49, 49));
pred_mean = [vec(nanmean(pred_scimlbnn[:, :, i], 1)) for i = 1:length(sensor_locs)]
plot!(pred_mean[25])
data_test[2:2:end,1,24]

nanquantile(x, q) = quantile(filter(!isnan, x), q)

Upper_sciml = [quantile(Array(pred_scimlbnn)[:, i,25], 0.95) for i = 1:length(Array(pred_scimlbnn)[1, :,1])]
Lower_sciml = [quantile(Array(pred_scimlbnn)[:, i,25], 0.05) for i = 1:length(Array(pred_scimlbnn)[1, :,1])]

Upper_bnn = [nanquantile(Array(pred_bnn)[:, i,46], 0.75) for i = 1:length(Array(pred_bnn)[1, :,1])]
Lower_bnn = [nanquantile(Array(pred_bnn)[:, i,46], 0.25) for i = 1:length(Array(pred_bnn)[1, :,1])]

begin
    gr()
    p1 =
        plot(sensor_locs, pred_mean_bnn[46], lw = 3, label = "BNN", color = "blue")
    plot!(sensor_locs, Lower_bnn, fillrange = Upper_bnn,
        fillalpha = 0.25, label = "0.9 C.I.",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "Deflection [mm]",
        yflip = true, color = nothing, fillcolor = "blue", xguidefontsize = 14, yguidefontsize = 14)
    #plot!(sensor_locs, pred_nn[:, 10], lw = 3, label = "NN", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", linestyle = :dash)
    plot!(sensor_locs, 1000 .* data_test[:, 1, 46], label = "Ground truth data", linestyle = :dashdot, lw = 2,
        legend = :bottomright, markerstrokewidth = 0, color = "green")
    hline!([0.0], color = "black", label = false)

    p2 =
        plot(sensor_locs, pred_mean[25], lw = 3, label = "BSciML", color = "blue")
    plot!(sensor_locs, Lower_sciml, fillrange = Upper_sciml,
        fillalpha = 0.25, color = nothing, fillcolor = "blue", label = "0.9 C.I.",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "",
        yflip = true, xguidefontsize = 14, yguidefontsize = 14)
    #plot!(sensor_locs, pred_sciml[:, 1, 1], lw = 3, label = "SciML", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)
    plot!(sensor_locs, 1000 .* data_test[:, 1, 25], label = "Ground truth data", linestyle = :dashdot, lw = 2,
        legend = :bottomright, markerstrokewidth = 0, color = "green")
        plot!(sensor_locs, tmp_lb[:,25,3], fillrange = tmp_ub[:,25,3],
        fillalpha = 0.25, color = nothing, fillcolor = "green", label = "0.9 C.I.Synth",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "",
        yflip = true, xguidefontsize = 14, yguidefontsize = 14)
    hline!([0.0], color = "black", label = false)

    title = plot(title = "Test sensor: 22.5 m. Test load lane: 1.25 m", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

    plot(title, p1, p2, layout = @layout([A{0.01h}; [B C]]), size = (1500, 500),#=  ylims = (-30,50), =#
        left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    #savefig("Results/Case 2/plots_training_region/inf_l_doubleplot_ll113.png")
end

function p_level(segment,sensor,level)
    # compute the level-quantile for a given segment and sensor
    # for the 2 bayesian models

    sciml_up = quantile(Array(pred_scimlbnn)[:, segment,sensor], 1 - ((1-level)/2))
    sciml_lb = quantile(Array(pred_scimlbnn)[:, segment,sensor], (1-level)/2)

    bnn_ub = nanquantile(Array(pred_bnn)[:, segment,sensor], 1 - ((1-level)/2))
    bnn_lb = nanquantile(Array(pred_bnn)[:, segment,sensor], (1-level)/2)

    return sciml_up, sciml_lb, bnn_ub, bnn_lb
end

#Create matrix that stores the coverage of each segment and sensor
p = zeros(49,49,3)
level = [0.1,0.5,0.9]
#= tmp_ub = zeros(49,49,3)
tmp_lb = zeros(49,49,3) =#
for l = 1:3
    for i=2:2:49 #only takes segments and sensors that were
                # not used for fitting
        for s = 2:2:49
        d = Normal(1000 .* data_test[i,1,s],50*maximum(abs.(data_test)))
        p[i,s,l] = cdf(d,p_level(i,s,level[l])[1]) - cdf(d,p_level(i,s,level[l])[2])
        #= tmp_ub[i,s,l] = quantile(d,1 - ((1-level[l])/2))
        tmp_lb[i,s,l] = quantile(d,(1-level[l])/2) =#


        end
    end
end

#uncertainty coverage mean per test sensor
p_c = p[2:2:49,2:2:49,:]
#save("Results/Accuracy_comparison/21_sensors/p_c_bsciml.jld", "p_c",p_c)
print("saved")
p_c_ub = zeros(24,3)
p_c_lb = zeros(24,3)

for i = 1:24
    for j = 1:3
        p_c_ub[i,j] = quantile(p_c[:,i,j],0.90)
        p_c_lb[i,j] = quantile(p_c[:,i,j],0.1)
    end
end

p_c_lb

p_c = load("Results/Accuracy_comparison/2_sensors/p_c_bnn.jld")["p_c"]
p_c_mean = mean(p_c, dims = 1)
c = ["red" "blue" "green"]
p = plot(title = "")
for i = 1:3
    p = plot!(sensor_locs[2:2:49],p_c_mean[1,:,i], w = 3,
    xlabel = "Test sensors [m]", color = c[i], legend = :outertopright, label = "p_coverage")
    #= plot!(sensor_locs[2:2:49], p_c_lb[:,i], fillrange = p_c_ub[:,i],
        fillalpha = 0.25, color = nothing, fillcolor = c[i], label = "",
        title = "", xlabel = "", yaxis = "",
        yflip = false, xguidefontsize = 14, yguidefontsize = 14) =#
    hline!(sensor_locs[2:2:49],[level[i]],color = c[i], label = "p_level = $(level[i])", ls = :dash, xlabel = "Test sensors [m]",
    ylabel = "Probability level", title = "BNN: 2 fitting sensors")
    display(p)
end

# compute the mean of all test sensors
# per case and plot
all_p_c_mean = zeros(25,24,3)
all_p_c_mean[1,:,:] = mean(load("Results/Accuracy_comparison/1_sensor/p_c_bnn.jld")["p_c"], dims = 1)

for i = 2:25
    all_p_c_mean[i,:,:] = mean(load("Results/Accuracy_comparison/" * "$i" * "_sensors/p_c_bnn.jld")["p_c"], dims = 1)
end
all_p_c_mean
p_c_mean_case = mean(all_p_c_mean, dims = 2)
p_c_case_ub = zeros(25,3)
p_c_case_lb = zeros(25,3)

for i = 1:25
    for j = 1:3
        p_c_case_ub[i,j] = quantile(all_p_c_mean[i,:,j],0.9)
        p_c_case_lb[i,j] = quantile(all_p_c_mean[i,:,j],0.1)
    end
end

p = plot(title = "")
for i = 1:3
    p = plot!(p_c_mean_case[:,1,i],w = 3,
     color = c[i], legend = :outertopright, label = "p_coverage")
    hline!(sensor_locs[2:2:49],[level[i]],color = c[i], label = "p_level = $(level[i])", ls = :dash)
    plot!(title = "BNN uncertainty coverage",xlabel = "# sensors",ylabel = "probability level", p_c_case_lb[:,i], fillrange = p_c_case_ub[:,i],
        fillalpha = 0.25, color = nothing, fillcolor = c[i], label = "",
        yflip = false, xguidefontsize = 14, yguidefontsize = 14)
    display(p)
end

# test
plot(sensor_locs[1:2:49],p_c_mean[1,:,2])
plot(sensor_locs[1:2:49],p_c[:,5,2], xlabel = "Longitudinal direction, x [m]")
p_c_mean[1,24,2]
d = Normal(1000 .* data_test[25,1,25],5 .*maximum(abs.(data_test)))
p_level(25,25,0.9)
a = cdf(d,p_level(25,25,0.9)[4])
plot(d)
quantile(Normal(0,1),0.95)
