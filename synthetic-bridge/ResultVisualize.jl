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
steps = 2.5 # size of steps for FEM
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

#NN

nn_params = load("Results/Case 2/nn_params.jld")["nn_params"]
pred_nn = zeros(length(sensor_locs),length(sensor_locs))
for i = 1:length(sensor_locs)
    pred_nn[:,i] = 1000 .* pure_bnn.(sensor_locs, load_lane_locs[1], [sensor_locs[i]], [nn_params])
end
plot(sensor_locs, pred_nn[:,10])

#Lower_sciml

#Load nn parameters and construct results
sciml_params = load("Results/Case 2/sciml_params.jld")["sciml_params"]

begin
    pred_sciml = Array{Float64,3}(undef, length(sensor_locs), length([load_lane_locs[1]]), length([sensor_locs[2]]))
    for i = 1:length([load_lane_locs[1]])
        for j = 1:length([sensor_locs[2]])
            # call neural network prediction * influence line
            pred_sciml[:, i, j] = 1000 .* broadcast(*, inf_l_test[:, 10], bnn.(sensor_locs,
                [load_lane_locs[1]]', x_middle, [sciml_params])[:, i])
        end
    end
end

begin
    plot(title = "Sciml no uncertainties",
        sensor_locs,
        pred_sciml[:, 1, 1],
        label = "sciml_prediction")
end

# BNN & BNN-SCIML

#load chain
chain_sciml = read("Results/Case 2/chain_sciml.jls", Chains)
chain_bnn = read("Results/Case 2/chain_bnn.jls", Chains)
group(chain_sciml, :std)
#predict model
test_model_sciml = bayesian_sciml(missing, sensor_locs, load_lane_locs[1],
    sensor_locs, num_params, inf_l_test);
test_model_bnn = bayesian_dd(missing, sensor_locs, load_lane_locs[1],
    sensor_locs);

#predict function

#if nan values use the following:
nanmean_1d(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean_1d, x, dims = y)

#bnn
pred_bnn = 1000 .* Array(predict(test_model_bnn, chain_bnn)) ./ 100
pred_bnn = reshape(pred_bnn, (1000, 25, 25))
pred_mean_bnn = [vec(nanmean(pred_bnn[:, :, i], 1)) for i = 1:length(sensor_locs)]
plot(pred_mean_bnn[10])

#bnn-sciml
pred_scimlbnn = reshape(1000 .* Array(predict(test_model_sciml, chain_sciml)), (1000, 25, 25));
pred_mean = [vec(nanmean(pred_scimlbnn[:, :, i], 1)) for i = 1:length(sensor_locs)]
plot!(pred_mean[10])

### Performance assessment
## Uncertainty
#multiple sensors sciml
std_scimlbnn_sensors = [nanmean_1d(
    (1 / (1000 - 1)) * sum([abs.(pred_mean[j] - Array(pred_scimlbnn)[i, :, j]) for i = 1:length(Array(pred_scimlbnn)[:, 1, 1])])
) for j = 1:length(sensor_locs)]
plot(sensor_locs, std_scimlbnn_sensors)

#1 sensor sciml
std_bsciml = (1 / (1000 - 1)) * sum([abs.(pred_mean[10] - Array(pred_scimlbnn)[i, :,10]) for i = 1:length(Array(pred_scimlbnn)[:, 1,1])])
mean(std_bsciml)
#multiple sensors bnn
std_ = zeros(length(sensor_locs), num_samples, length(sensor_locs))
for j = 1:25
    for i = 1:1000
        std_[:,i,j] = abs.(pred_mean_bnn[j] - Array(pred_bnn)[i, :,j])
    end
end
std_bnn_sensors = nanmean(nanmean(std_,2),1)[:]

plot(sensor_locs, std_scimlbnn_sensors, xlabel = "Longitudinal direction, xₛ [m]", ylabel = "Std",
 title = "Std per test sensor", label = "BSciML"#= , yaxis = :log =#)
plot!(sensor_locs, std_bnn_sensors, label = "BNN"#= , yaxis =:log =#)

#save("Results/Accuracy_comparison/25_sensors/std_bnn_sensors.jld", "std_bnn_sensors",std_bnn_sensors)

#Load all std scenarios
std_matrix = zeros(length(sensor_locs), length(sensor_locs))
for i = 2:25
    std_matrix[:,i] = load("Results/Accuracy_comparison/" * "$i" * "_sensors/std_bnn_sensors.jld")["std_bnn_sensors"]
end

STD = nanmean(std_matrix,1)

#1 sensor bnn
std_bnn_ = zeros(25,1000)

for i = 1:1000
    std_bnn[:,i] = abs.(pred_mean_bnn[2] - Array(pred_bnn)[i, :,2])
end

std_bnn = nanmean(std_bnn,2)


begin
    gr()
    plot(sensor_locs, variance_scimlbnn4, ylabel = " Standard Deviation [mm]", label = "BSciML", #= yaxis = :log, =#
        xlabel = "Longitudinal direction, x[m]", title = "Probabilistic models standard deviations")
    plot!(sensor_locs, variance_bnn, label = "BNN") #= yaxis = :log, =#
    vline!([0.0, 20.0, 40.0, 60.0], label = "", ls = :dash)
    #savefig("Results/Case 2/plots_test_region/loss_scimlbnnll3s13.png")
end

## RMSE
# Single sensor: calculates the loss per node between the averaged predictions and the synthetic data
mean_loss_scimlbnn = abs.(pred_mean[10] - 1000 .* data_test[:, 1, 10])
mean_loss_bnn = abs.(pred_mean_bnn[10] - 1000 .* data_test[:, 1, 10])
mean(mean_loss_scimlbnn)
mean(mean_loss_bnn)
#All the sensors: calculates the RMSE for every sensor.
RMSE_scimlbnn = [nanmean_1d(abs.(pred_mean[i] - 1000 .* data_test[:, 1, i])) for i = 1:length(sensor_locs)]
RMSE_bnn = [nanmean_1d(abs.(pred_mean_bnn[i] - 1000 .* data_test[:, 1, i])) for i = 1:length(sensor_locs)]
RMSE_nn = [nanmean_1d(abs.(pred_nn[:,i] - 1000 .* data_test[:, 1, i])) for i = 1:length(sensor_locs)]

#save("Results/Accuracy_comparison/25_sensors/RMSE_bnn.jld", "RMSE_bnn",RMSE_bnn)

#RMSE_bnn = load("Results/Accuracy_comparison/2_sensors/RMSE_bnn.jld")["RMSE_bnn"]
plot(sensor_locs, RMSE_scimlbnn, xlabel = "Longitudinal direction, x [m]", ylabel = "RMSE",
 title = "RMSE per test sensor", label = "BSciML")
plot!(sensor_locs, RMSE_bnn, label = "BNN")

#Load all RMSEs scenarios
RMSE_matrix = zeros(length(sensor_locs), length(sensor_locs))
for i = 2:25
    RMSE_matrix[:,i] = load("Results/Accuracy_comparison/" * "$i" * "_sensors/RMSE_bnn.jld")["RMSE_bnn"]
end
RMSE = nanmean(RMSE_matrix,1)

## Compare std and RMSE
plot(STD' .- 0.05*maximum(abs.(data_test)), label = "Epistemic uncertainty",
xlabel = "Number of sensors", ylabel = "RMSE/Std [mm]", title = "Uncertainty coverage")
plot!(RMSE', label = "RMSE")

mean_loss_bnn = abs.(pred_mean_bnn - 1000 .* data_test[:, 2, 10])

begin
    gr()
    plot(sensor_locs, mean_loss_scimlbnn, ylabel = " Mean Error [mm]",#=  yaxis = :log, =#
        xlabel = "Longitudinal direction, x[m]", label = "SciML-BNN")
    plot!(sensor_locs, mean_loss_bnn, label = "BNN")#=  yaxis = :log, =#
    vline!([0.0, 20.0, 40.0, 60.0], label = "", ls = :dash)
    #savefig("Results/Case 8/plots_test_region/meanloss_scimlbnnll1s13.png")

end

#Compare losses.
begin
    gr()

    p1 = plot(sensor_locs, variance_scimlbnn2, ylabel = " Variance [mm]", label = "5 Load Lanes", #= yaxis = :log, =#
        xlabel = "Longitudinal direction, x [m]")
    plot!(sensor_locs, variance_scimlbnn7, label = "2 Load Lanes") #= yaxis = :log, =#
    plot!(sensor_locs, variance_scimlbnn4, label = "1 Load Lane") #= yaxis = :log, =#
    vline!([0.0, 20.0, 40.0, 60.0], label = "", ls = :dash)

    p2 = plot(sensor_locs, mean_loss_scimlbnn2, ylabel = " Mean Error [mm]",#=  yaxis = :log, =#
        xlabel = "Longitudinal direction, x [m]", label = "5 Load Lanes")
    plot!(sensor_locs, mean_loss_scimlbnn7, label = "2 Load Lane")#=  yaxis = :log, =#
    plot!(sensor_locs, mean_loss_scimlbnn4, label = "1 Load Lanes") #= yaxis = :log, =#
    vline!([0.0, 20.0, 40.0, 60.0], label = "", ls = :dash)

    title = plot(title = "Effect of adding load lanes", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

    plot(title, p1, p2, layout = @layout([A{0.01h}; [B C]]), size = (1200, 400), left_margin = 6Plots.mm, bottom_margin = 7Plots.mm)

    #savefig("Results/Case 7.1/plots_test_region/double_losses_scimlbnnll2s10.png")


end

# Find Quantiles

#if nan values use:
nanquantile(x, q) = quantile(filter(!isnan, x), q)

Upper_sciml = [quantile(Array(pred_scimlbnn)[:, i,10], 0.95) for i = 1:length(Array(pred_scimlbnn)[1, :,1])]
Lower_sciml = [quantile(Array(pred_scimlbnn)[:, i,10], 0.05) for i = 1:length(Array(pred_scimlbnn)[1, :,1])]

Upper_bnn = [nanquantile(Array(pred_bnn)[:, i,10], 0.95) for i = 1:length(Array(pred_bnn)[1, :,1])]
Lower_bnn = [nanquantile(Array(pred_bnn)[:, i,10], 0.05) for i = 1:length(Array(pred_bnn)[1, :,1])]


#PLOTTING

begin
    gr()
    p1 =
        plot(sensor_locs, pred_mean_bnn[10], lw = 3, label = "BNN", color = "blue")
    plot!(sensor_locs, Lower_bnn, fillrange = Upper_bnn,
        fillalpha = 0.25, label = "0.9 C.I.",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "Deflection [mm]",
        yflip = true, color = nothing, fillcolor = "blue", xguidefontsize = 14, yguidefontsize = 14)
    plot!(sensor_locs, pred_nn[:, 10], lw = 3, label = "NN", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", linestyle = :dash)
    plot!(sensor_locs, 1000 .* data_test[:, 1, 10], label = "Ground truth data", linestyle = :dashdot, lw = 2,
        legend = :bottomright, markerstrokewidth = 0, color = "green")
    hline!([0.0], color = "black", label = false)

    p2 =
        plot(sensor_locs, pred_mean[10], lw = 3, label = "BSciML", color = "blue")
    plot!(sensor_locs, Lower_sciml, fillrange = Upper_sciml,
        fillalpha = 0.25, color = nothing, fillcolor = "blue", label = "0.9 C.I.",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "",
        yflip = true, xguidefontsize = 14, yguidefontsize = 14)
    plot!(sensor_locs, pred_sciml[:, 1, 1], lw = 3, label = "SciML", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)
    plot!(sensor_locs, 1000 .* data_test[:, 1, 10], label = "Ground truth data", linestyle = :dashdot, lw = 2,
        legend = :bottomright, markerstrokewidth = 0, color = "green")
    hline!([0.0], color = "black", label = false)

    title = plot(title = "Test sensor: 22.5 m. Test load lane: 1.25 m", grid = false, framestyle = nothing, showaxis = false, xticks = false, yticks = false)

    plot(title, p1, p2, layout = @layout([A{0.01h}; [B C]]), size = (1500, 500),#=  ylims = (-30,50), =#
        left_margin = 10Plots.mm, bottom_margin = 7Plots.mm, titlefontsize = 14) #= link =:y, =#
    #savefig("Results/Case 2/plots_training_region/inf_l_doubleplot_ll113.png")
end

### Influence surface

#Theoretical
inf_sur_gt = repeat(1000 .* inf_l_test[:, 10], 1, length(load_lane_locs)) .* ground_truths

## predicted deflection surface
#sciml
# find predicted tld
tld_pred = zeros(length(chain_sciml), length(sensor_locs), length(load_lane_locs))
for i = 1:length(chain_sciml)

    tld_pred[i, :, :] = bnn.(sensor_locs, load_lane_locs', x_middle,
        [Array(chain_sciml)[i, 1:end]])

end

tld_pred_mean = nanmean(Array(tld_pred), 1)[1, :, :]

#pred influence surface
inf_sur_sciml = repeat(1000 .* inf_l_test[:, 10], 1, length(load_lane_locs)) .*
                tld_pred_mean


#bnn.

inf_sur_bnn = zeros(length(chain_sciml), length(sensor_locs), length(load_lane_locs))
for i = 1:length(chain_bnn)

    inf_sur_bnn[i, :, :] = 10 .* pure_bnn.(sensor_locs, load_lane_locs', [sensor_locs[10]],
        [Array(chain_bnn)[i, 1:end]]) #The factor of 10 keeps units

end

inf_sur_bnn = nanmean(Array(inf_sur_bnn), 1)[1, :, :]

begin
    plotlyjs()

    p1 = plot(inf_sur_gt', st = :contourf, fill = fill,
        color = cgrad(:diverging, [0.25, 0.75]),
        xticks = get_ticks(5, sensor_locs),
        yticks = get_ticks(1, load_lane_locs),
        ylabel = "", xlabel = "Longitudinal direction, x [m]",
        xguidefontsize = 14,
        colorbar = :none,
        clims = (-maximum(inf_sur_sciml), maximum(inf_sur_sciml)))
    scatter!([10], [1], label = "test sensor", color = :yellow)
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([1], color = :yellow, legend = false)

    p2 = plot(inf_sur_sciml', st = :contourf, fill = fill,
        color = cgrad(:diverging, [0.25, 0.75]),
        xticks = get_ticks(5, sensor_locs),
        yticks = get_ticks(1, load_lane_locs),
        ylabel = "",        #= xlabel = "x[m]", =#
        colorbar_title = " \nDeflection [mm]",
        right_margin = 4Plots.mm,
        colorbar = :right,
        clims = (-maximum(inf_sur_sciml), maximum(inf_sur_sciml)))
    scatter!([10], [1], label = "Test sensor", color = :yellow)
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([1], color = :yellow, legend = false)

    p3 = plot(inf_sur_bnn', st = :contourf, fill = fill,
        color = cgrad(:diverging, [0.25, 0.75]),
        xticks = get_ticks(5, sensor_locs),
        yticks = get_ticks(1, load_lane_locs),
        ylabel = "Transverse direction, z [m]",
        yguidefontsize = 14,       #= xlabel = "x[m]", =#
        right_margin = 4Plots.mm,
        colorbar = :none,
        clims = (-maximum(inf_sur_sciml), maximum(inf_sur_sciml)))
    scatter!([10], [1], label = "Test sensor", color = :yellow)
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([1], color = :yellow, legend = false)

    title = plot(title = "", grid = false, showaxis = false, xticks = false, yticks = false)

    plot(title, p3, p1, p2, layout = @layout([A{0.001h}; [B C D]]), title = ["Influence surface for sensor at x = 22.5 m" "BNN Prediction" "Ground Truth Influence Surface" "SciML-BNN Prediction"], size = (1500, 500),
        left_margin = 7Plots.mm, bottom_margin = 4Plots.mm,
        legendfontvalign = :center,
        titlefontsize = 14)



    #savefig("Results/Case 2/plots_training_region/inf_surface-s13_pdf.pdf")

end

#tld uncertainty surface
variance_matrix = zeros(length(sensor_locs), length(load_lane_locs))
for i = 1:length(sensor_locs)
    for j = 1:length(load_lane_locs)
        variance_matrix[i, j] = (1 / 999) .* sum(abs2.(tld_pred[:, i, j] .- tld_pred_mean[i, j]))
    end
end



#contour plot for the mean predicted transverse load dist.
begin
    plotlyjs()
    plt1 = plot(tld_pred_mean', st = :contour, fill = fill,
        vline = ([0, 20, 40, 60]),
        yticks = (1:5, [0, 1.25, 2.50, 3.75, 5]),
        xticks = get_ticks(5, sensor_locs),
        xlabel = "Longitudinal direction, xₚ [m]",
        xguidefontsize = 14,
        colorbar_title = " Load fraction",
        clims = (minimum(ground_truths), maximum(ground_truths)))
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([1, 2, 3, 4, 5], color = :yellow, legend = false)


    plt2 = plot(ground_truths', st = :contour, fill = fill,
        yticks = (1:5, load_lane_locs),
        xticks = get_ticks(5, sensor_locs),
        xlabel = "Longitudinal direction, xₚ [m]",
        xguidefontsize = 14,
        yguidefontsize = 14,
        ylabel = "Transverse direction, zₚ [m]",
        colorbar_title = " Load fraction",
        clims = (minimum(ground_truths), maximum(ground_truths)))
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([1, 2, 3, 4, 5], color = :yellow, legend = false)


    plt3 = plot(title = "", grid = false, showaxis = false, xticks = false, yticks = false) #= framestyle = nothing, =#


    plot(plt3, plt2, plt1, layout = @layout([A{0.001h}; [B C]]), size = (1500, 500), title = ["Fitting sensor: x = 30 m, Load lane: (0, 1.25, 2.5, 3.75, 5) m" "TLD ground truth" "TLD prediction"],
        bottom_margin = 7Plots.mm, left_margin = 7Plots.mm,
        titlefontsize = 14)

    #savefig("Results/Case 4/plots_training_region/TLD_contour.png")
end

begin
    plotlyjs()
    plot(variance_matrix', st = :contour, fill = fill,
        title = "TLD variance surface",
        clims = (0, 0.3),
        color = cgrad(:diverging, [0.25, 0.75], rev = true),
        yticks = (1:5, load_lane_locs),
        xticks = get_ticks(5, sensor_locs),
        xlabel = "Longitudinal direction, x[m]",
        xguidefontsize = 14,
        ylabel = "Transverse direction, z[m]",
        yguidefontsize = 14,
        titlefontsize = 14,
        colorbar_title = "Variance", #= size = (1000,333), =#
        left_margin = 7Plots.mm, bottom_margin = 7Plots.mm)
    vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([0, 1.25, 2.5, 3.75, 5], get_ticks(1, load_lane_locs), linestyle = :dash, label = :none,
        color = :black)
    hline!([2, 4], color = :yellow, legend = false)
    #savefig("Results/Case 4/plots_training_region/TLD_Uncertainty.png")
end


#uncertainty
begin
    plt = plot(sensor_locs, ground_truths[:, 1], label = "ground_truth", w = 4, ylim = (-1, 1.5))
    plot!(plt, sensor_locs, mean_pred_f[:, 1], w = 4,
        label = "predicted mean")

    for k = 1:10:num_samples
        plot!(plt, sensor_locs, Array(pred_f)[k, :, 1], alpha = 1,
            color = "#BBBBBB", legend = false)
    end

    vline!([0.0, 20, 40, 60], label = "supports", w = 2,
        color = "green")
    vline!(sensor_locs, alpha = 0.5, label = "nodes")

end

#Quantiles

function Quantiles(load_lane_id)
    Upper = [nanquantile(Array(tld_pred)[:, i, load_lane_id], 0.95) for i = 1:length(Array(tld_pred)[1, :, 1])]
    Lower = [nanquantile(Array(tld_pred)[:, i, load_lane_id], 0.05) for i = 1:length(Array(tld_pred)[1, :, 1])]
    return Upper, Lower
end

Upper1, Lower1 = Quantiles(1)
Upper2, Lower2 = Quantiles(2)
Upper3, Lower3 = Quantiles(3)
Upper4, Lower4 = Quantiles(4)
Upper5, Lower5 = Quantiles(5)

#plot uncertainty bands
begin
    plotlyjs()
    p1 =
        plot(sensor_locs, tld_pred_mean[:, 1], label = "")
    plot!(sensor_locs, Lower1, fillrange = Upper1,
        fillcolor = "blue", label = "",
        title = "", xlabel = "", yaxis = "TLD", color = nothing,
        fillalpha = 0.3)
    plot!(sensor_locs, ground_truths[:, 1], label = "", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)

    p2 =
        plot(sensor_locs, tld_pred_mean[:, 2], label = "")
    plot!(sensor_locs, Lower2, fillrange = Upper2,
        fillcolor = "blue", label = "",
        title = "", xlabel = "", yaxis = "", color = nothing,
        fillalpha = 0.3)
    plot!(sensor_locs, ground_truths[:, 2], label = "", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)

    p3 =
        plot(sensor_locs, tld_pred_mean[:, 3], label = "")
    plot!(sensor_locs, Lower3, fillrange = Upper3,
        fillcolor = "blue", label = "",
        title = "", xlabel = "", yaxis = "TLD", color = nothing,
        fillalpha = 0.3)
    plot!(sensor_locs, ground_truths[:, 3], label = "", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)

    p4 =
        plot(sensor_locs, tld_pred_mean[:, 4], label = "")
    plot!(sensor_locs, Lower4, fillrange = Upper4,
        fillcolor = "blue", label = "",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "", color = nothing,
        fillalpha = 0.3)
    plot!(sensor_locs, ground_truths[:, 4], label = "", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)

    p5 =
        plot(sensor_locs, tld_pred_mean[:, 5], label = "Predicted TLD")
    plot!(sensor_locs, Lower5, fillrange = Upper5,
        fillcolor = "blue", label = "0.9 Credible interval",
        title = "", xlabel = "Longitudinal direction, x [m]", yaxis = "TLD", color = nothing,
        fillalpha = 0.3)
    plot!(sensor_locs, ground_truths[:, 5], label = "Ground truth", color = "red")
    vline!([0.0, 20, 40.0, 60.0], label = "", ls = :dash)

    p6 = plot(title = "")

    plot(p1, p2, p3, p4, p5, p6, layout = grid(3, 2), size = (1000, 500), legend = :bottom,
        title = ["z = 0 m" "z = 1.25 m" "z = 2.5 m" "z = 3.75 m" "z = 5 m" ""],
        titlefontsize = 11)

    #savefig("Results/Case 2/plots_training_region/TLD_uncertainty.svg")
end


plot(ground_truths', st = :contour, fill = fill,
    yticks = (1:5, load_lane_locs),
    xticks = get_ticks(5, sensor_locs),
    xlabel = "Longitudinal direction, xₚ[m]",
    xguidefontsize = 14,
    yguidefontsize = 14,
    ylabel = "Transverse direction, zₚ[m]",
    colorbar_title = " Load fraction",
    clims = (minimum(ground_truths), maximum(ground_truths)),
    size = (1000, 500),
    title = "TLD ground truth",
    left_margin = 7Plots.mm)
vline!([0, 20, 40, 60], get_ticks(8, sensor_locs), linestyle = :dash, label = :none,
    color = :black)
