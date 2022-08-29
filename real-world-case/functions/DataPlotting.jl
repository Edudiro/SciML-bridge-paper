module DataUtils

export loadsensordata, Il_obj, dd_pred_il_plot, sciml_pred_il_plot,
        create_prediction_error_plot, multi_scale, findall_signchange

#= include(abspath(joinpath("continuous_girder", "analysis", "Configs.jl")))
using .Configs: ANALYSIS_DIR =#

include("Utils.jl")
using .Utils: relative

using CSV: File
using Statistics: mean
using DataFrames: DataFrame, filter, levels
using Interpolations: LinearInterpolation
using Plots: plot, plot!, vline!, hline!, savefig, annotate!, text, scatter!
using StatsPlots
using CategoricalArrays: CategoricalArray, levels!
using Printf: @sprintf
using Turing

"""
Load the measurements from a particular sensor
"""
function loadsensordata_sciml(sensor_name, path, girder)

    support_xs = girder["support_xs"] / 1e3
    node_xs = girder["node_xs"] / 1e3

    meas_df = DataFrame(File(path, datarow=3))

    # Filter data by sensor name
    meas_df = filter(:position => ==(sensor_name), meas_df)

    # Interpolate the measurements over a shared x grid
    #Hack
    corrected_node_xs = copy(node_xs[2:end-1])
    node_xs = corrected_node_xs
    num_meas_x = length(node_xs)
    meas_xs = LinRange(support_xs[1], support_xs[end], num_meas_x)

    truck_positions = levels(meas_df.truck_position)
    meas_stress = Array{Float64}(undef, num_meas_x, length(truck_positions))
    for (ii, truck_position) in enumerate(truck_positions)
        meas_df_ii = filter(:truck_position => ==(truck_position), meas_df)
        meas_x_ii = meas_df_ii.x
        meas_stress_ii = meas_df_ii.stress * 1e3 # MPa -> kN/m2
        itp = LinearInterpolation(meas_x_ii, meas_stress_ii)
        # Hack
        meas_stress[:, ii] = itp(node_xs)
        #meas_stress[:, ii] = itp(meas_xs)
    end

    right_meas_stress_il = meas_stress[:, findfirst("right" .== truck_positions)]
    left_meas_stress_il = meas_stress[:, findfirst("left" .== truck_positions)]
    return right_meas_stress_il, left_meas_stress_il
end

function loadsensordata(sensor_name, meas_path, girder)

    support_xs = girder["support_xs"] / 1e3
    node_xs = girder["node_xs"] / 1e3

    meas_df = DataFrame(File(meas_path, datarow=3))

    # Filter data by sensor name
    meas_df = filter(:position => ==(sensor_name), meas_df)

    # Interpolate the measurements over a shared x grid
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
    return right_meas_stress_il, left_meas_stress_il
end

"""
Influence line object stores data from influence line prediction.
"""

struct Il_obj
    meas_xs::Vector{Float64}
    support_xs::Vector{Float64}
    right_meas_stress_il::Vector{Float64}
    right_pred_stress_il::Vector{Float64}
    left_meas_stress_il::Vector{Float64}
    left_pred_stress_il::Vector{Float64}
end

function _pred_il_plot(il, pred_sensor_name, pred_sensor_position,
    data_sensor_positions, lims=[-20, 20], test_type="SciML")

    max_i_left = abs.(il.left_meas_stress_il .- il.left_pred_stress_il) |> x->findfirst(isequal(maximum(x)), x)
    max_i_right = abs.(il.right_meas_stress_il .- il.right_pred_stress_il) |> x->findfirst(isequal(maximum(x)), x)

    lw = 3
    y_scale = 1e-3 # kN/m2 -> MPa
        begin
        pp = plot(title=@sprintf("Predicted influence line at %s", pred_sensor_name),
        xlabel="Position of the first axle [m]", ylabel="Stress [MPa]",
        legend=:topright, ylims = lims)
        # Right truck loading
        plot!(il.meas_xs, il.right_meas_stress_il * y_scale,
        linewidth=lw, color=1, linealpha=0.5, label="right, measured")
        plot!(il.meas_xs, il.right_pred_stress_il * y_scale,
        linewidth=lw, color=1, label="right, "*test_type, ls=:dot)
        # Left truck loading
        plot!(il.meas_xs, il.left_meas_stress_il * y_scale,
        linewidth=lw, color=2, linealpha=0.5, label="left, measured")
        plot!(il.meas_xs, il.left_pred_stress_il * y_scale,
        linewidth=lw, color=2, label="left, "*test_type, ls=:dot)

        vline!(data_sensor_positions, color="blue", ls=:dashdot, label=false)
        #for  sensor_position in data_sensor_positions
            #annotate!(sp=1, relative(pp[1], sensor_position / il.support_xs[end], 0.05)...,
            #text("data", 8, :blue, :left, :bottom, rotation=90))
        #end

        vline!([pred_sensor_position], color="red", ls=:dashdot, label=false, linewidth = lw-1)
        #annotate!(sp=1, relative(pp[1], pred_sensor_position / il.support_xs[end], 0.05)...,
        #text("predict", 8, :red, :left, :bottom, rotation=90))

        plot!([1], [0], linestyle = :dashdot, label = "measurement", color = "blue")
        plot!([1], [0], linestyle = :dashdot, label = "prediction", color = "red", linewidth = lw-1)

        plot!([il.meas_xs[max_i_left], il.meas_xs[max_i_left]],
            [il.left_meas_stress_il[max_i_left], il.left_pred_stress_il[max_i_left]]* y_scale,
            ls = :solid, color = "red", label = "max difference", shape = :hline, ms=3, msc = "red", msw = 1.5)
        plot!([il.meas_xs[max_i_right], il.meas_xs[max_i_right]],
        [il.right_meas_stress_il[max_i_right], il.right_pred_stress_il[max_i_right]]* y_scale,
        ls = :solid, color = "red", label = "", shape = :hline, ms=3, msc = "red", msw = 1.5)

    end
return pp
end





function dd_pred_il_plot(il, pred_sensor_name, pred_sensor_position,
    data_sensor_positions, lims=[-20, 20])

    return _pred_il_plot(il, pred_sensor_name, pred_sensor_position,
                            data_sensor_positions, lims, "DD")
end

function sciml_pred_il_plot(il, pred_sensor_name, pred_sensor_position,
    data_sensor_positions, lims=[-20, 20])

    return _pred_il_plot(il, pred_sensor_name, pred_sensor_position,
                            data_sensor_positions, lims, "SciML")
end




function create_prediction_error_plot(r_pred_f, l_pred_f, sensorinfo, all_stress_il, data_sensor_names, meas_path, girder, fe_mode)
_sensornames = collect(keys(sensorinfo)) |> sort! |> x->circshift(x, -1)
sensor_pos = [sensorinfo[name] for name in _sensornames]

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

r_meas = []
l_meas = []

for i = 1:length(_sensornames)
    (right_meas_stress_il, left_meas_stress_il) = loadsensordata(_sensornames[i], meas_path, girder)
    push!(r_meas, right_meas_stress_il)
    push!(l_meas, left_meas_stress_il)
end



# ------------------------------------------------------
# MAKE PREDICTIONS & GET ERRORS
# ------------------------------------------------------
r_avg_error = zeros(length(_sensornames))
l_avg_error = zeros(length(_sensornames))
r_max_error = zeros(length(_sensornames))
l_max_error = zeros(length(_sensornames))



for i = 1: length(_sensornames)
    r_pred = r_pred_f(all_stress_il[i], sensor_pos[i])
    l_pred = l_pred_f(all_stress_il[i], sensor_pos[i])

    r_avg_error[i] = mean(abs.(r_pred .- r_meas[i])) * 1e-3 # MPa
    l_avg_error[i] = mean(abs.(l_pred .- l_meas[i])) * 1e-3 # MPa

    r_max_error[i] = maximum(abs.(r_pred .- r_meas[i])) * 1e-3 # MPa
    l_max_error[i] = maximum(abs.(l_pred .- l_meas[i])) * 1e-3 # MPa

end


# ------------------------------------------------------
# PLOT BAR ERROR CHART
# ------------------------------------------------------
_title = @sprintf("Error of predictions, data at %s", join(data_sensor_names, ", "))

if fe_mode == true
    _title = "Error of predictions, baseline FE model"
elseif length(data_sensor_names) == 9
    _title = "Error of predictions, data at all sensors"
elseif length(data_sensor_names) >= 7
    missing_sensors = symdiff(_sensornames, data_sensor_names)
    _title = @sprintf("Error of predictions, data at all sensors except %s", join(missing_sensors, ", "))
end
#ctg = repeat(["Right", "Max Error", "Left"], inner = length(_sensornames))

ctg = CategoricalArray(repeat(["Right RMSE", "Left RMSE"], inner = length(_sensornames)))
levels!(ctg, ["Right RMSE", "Left RMSE"])


function superscriptnumber(i::Int)
    if i < 0
        c = [Char(0x207B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        if d == 0 push!(c, Char(0x2070)) end
        if d == 1 push!(c, Char(0x00B9)) end
        if d == 2 push!(c, Char(0x00B2)) end
        if d == 3 push!(c, Char(0x00B3)) end
        if d > 3 push!(c, Char(0x2070+d)) end
    end
    return join(c)
end

y_limits = [0.1,1000]

function generatelogticks(lims)
ticklocs = Float64[]
ticknames = String[]
a = log10.(lims)
alin = round.(Int, collect(a[1]:1:a[end]))
delta = a[end] -a[1]

for i = 1:delta + 1
    base = [1, 2, 5]
    if i == delta + 1
        base = [1]
    end

    resloc = base * 10.0^alin[Int(i)]
    resname = string.(base) .* "·10" .* superscriptnumber(alin[Int(i)])
    push!(ticklocs, resloc...)
    push!(ticknames, resname...)
end

return (ticklocs, ticknames)
end




gb = groupedbar([r_avg_error l_avg_error ], group = ctg, bar_position = :dodge, bar_width=0.8,
            xticks=(1:length(_sensornames), _sensornames), ylabel = "RMSE", legend = :outerright, size = (750, 400),
            title = _title, color = [1 2], seriesalpha = 1.0, ylims = y_limits, yticks = generatelogticks(y_limits), yscale=:log10)

            for i = 1:length(_sensornames)
                x_l = i + 0.2
                x_r = i - 0.2

                y1_l = l_avg_error[i]
                y1_r = r_avg_error[i]

                y2_l = l_max_error[i]
                y2_r = r_max_error[i]

                plot!(gb, [x_l, x_l], [y1_l, y2_l], color = :black, label = "")
                plot!(gb, [x_r, x_r], [y1_r, y2_r], color = :black, label = "")

                scatter!(gb, [x_l], [y2_l], color= :black, shape = :hline, label = "", msw =1.5)
                scatter!(gb, [x_r], [y2_r], color= :black, shape = :hline, label = "", msw =1.5)

            end

            plot!(gb,[1], [0], linestyle = :solid, label = "max error", color = "black")

            hline!(gb, [2], color="#31a354", ls=:dashdot, label="accepted level", linewidth = 2)
return gb
        end





        function multi_scale(xs, zerovals_ind, new_zerovals_ind)

            zerovals = xs[zerovals_ind]
            new_zerovals = xs[new_zerovals_ind]



            xs_corr = copy(xs)
            for i = 2:length(zerovals)

                ind_min = zerovals_ind[i-1] # Index of current minimum
                next_xs = @view(xs_corr[ind_min:end]) # Values to modify

                oldmin = xs_corr[ind_min] # Current minimum
                oldmax = xs_corr[zerovals_ind[i]] # Current maximum
                newmax = new_zerovals[i] # Future maximum


                scale_ratio = (newmax - oldmin) / (oldmax - oldmin)
                next_xs[:] .-= oldmin
                next_xs[:] .*= scale_ratio
                next_xs[:] .+= oldmin

                #display(plot!(xs_corr, ys, label = @sprintf("Fit %d", i-1)))

            end
            return xs_corr
        end


    function findall_signchange(vals, delay = 0, maxlevel = typemax(Int64))
        tol = eps(Float32)
        new_vals = copy(vals)
        for i =1:length(new_vals)

            if abs(new_vals[i]) <= tol
                new_vals[i] = 0.0
            end
        end

        vals_sign = sign.(new_vals)
        changelist = Int64[]
        d = delay
        for i = 1:length(vals_sign)
            d += 1
            if d >= delay && i <= maxlevel

                sign_i = vals_sign[i]
                if sign_i == 0.0
                    push!(changelist, i)
                    d = 0

                elseif i > 1 && !(vals_sign[i-1] == 0.0) && sign_i != vals_sign[i-1]
                    push!(changelist, i)
                    d = 0

                #elseif i == maxlevel
                    #push!(changelist, i)

                end
            end


        end
        return changelist
    end
end
