
function data_generator(load_lane_offsets::AbstractVector, sensor_locs::Vector, zdim::Int, steps)

    # Element properties
    E = 200e9    # [Pa]
    Iy = 8.2e-3  # [m^4]
    P = 500.0e3  # [N]
    A = 2.0e-3   # [m^2]

    ep = [E A Iy]'

    # Bridge x-axis geometry
    elem_geo = [0.0, 20.0, 40.0, 60.0]
    L = last(elem_geo)
    step = steps# size of steps
    spans = [20.0, 20.0, 20.0]
    support_pos = [0, cumsum(spans)...]
    x_middle = support_pos[end] / 2
    max_span = maximum(spans)

    # Bridge z-axis geometry
    bridge_width = 5.0
    num_load_cases = length(load_lane_offsets)

    # Build simple model
    (K, f, bc, Edof) = build_girder_bridge(elem_geo, support_pos, ep)


    # Add custom element properties if applicable
    # EP is a vector of arrays with element properties
    # EP = [ep,ep] for example
    # Default is EP = ep
    EP = ep

    # Subdivide model
    (K2, f2, newBc, NewEdof, segments) = rebuild_linear_system(L, step, elem_geo, EP, bc, Edof, f, 2)

    # Create ranges in x and z direction
    xr = range(0, stop = L, length = length(segments))
    zr = range(0, stop = bridge_width, length = zdim)


    # ------------------------------------------------------
    # CREATE GROUND TRUTH
    # ------------------------------------------------------

    # Get index of current sensor
    sensor_id = (x -> (findfirst(isequal(x), segments) - 1) * 2 + 1).(sensor_locs)


    # Ground truth abiding to boundary condition
    function tlv_f(x, z)
        idx = support_pos[1:(end-1)] .<= x .<= support_pos[2:end]
        span = spans[idx][1]
        cs_span = support_pos[1:(end-1)][idx][1]
        fx = 1 - 0.5 * span / max_span * sin(π * (x - cs_span) / span)
        fz = 1 - sqrt(z / (2 * bridge_width))
        return fx * fz
    end

    # ------------------------------------------------------
    # COMPUTE VERTICAL TRANSLATION INFL. LINE FOR SENSOR
    # ------------------------------------------------------
    function translation_at_sensor(sensor_id, load, load_id)
        f = zeros(size(f2))
        f[load_id] = load
        (u, fb) = solveq(K2, f, newBc)
        return u[sensor_id]
    end


    # Initialize unit influence line
    unit_inf_line = zeros(length(segments), length(sensor_locs))

    # Fill unit influence line for training
    for i = 1:length(sensor_locs)
        unit_inf_line[:, i] =2 .* translation_at_sensor.(sensor_id[i], P, 1:2:length(segments)*2)
    end

    # Create unit influence line for validation (refactor this later in a good way.)
    all_unit_inf_lines = zeros(length(segments), length(segments))
    all_max_inf_lines = zeros(length(segments), length(segments))
    for i = 1:length(segments)
        _unit_infline = 2 .* translation_at_sensor.(1 + 2 * (i - 1), P, 1:2:length(segments)*2)
        all_unit_inf_lines[:, i] = _unit_infline
        all_max_inf_lines[:, i] = _unit_infline .* tlv_f.(segments, zeros(length(segments)))
    end

    # ------------------------------------------------------
    # CREATE TRAINING DATA
    # ------------------------------------------------------

    # load carried by the girder
    func_inf_lines = tlv_f.(segments, load_lane_offsets')

    meas_inf_line = zeros(length(segments), length(load_lane_offsets), length(sensor_locs))

    for i = 1:length(sensor_locs)
        meas_inf_line[:, :, i] = unit_inf_line[:, i] .* func_inf_lines
    end

    data = meas_inf_line
    noisydata = zeros(size(data))

    #noise = % of deflection
    #= ϵ = zeros(size(data))
    for i = 1:length(load_lane_offsets)
        for j = 1:length(sensor_locs)
            ϵ[:, i, j] = rand(MvNormal(zeros(length(data[:, i, j])), 0.05 .* data[:, i, j]))
        end
    end
    noisydata = data + ϵ =#

    #noise = % max abs(deflection)
    Random.seed!(1234)
    ϵ = Normal(0.0,0.05*maximum(abs.(data)))
    noisydata = data .+ rand(ϵ,size(data))

    #Interpolations

    #= itp_x = collect(range(0,L,length = num_obs))
    itp_data = zeros(length(itp_x), length(load_lane_offsets), length(sensor_locs))
    for i = 1:length(load_lane_offsets)
        for j = 1:length(sensor_locs)
            itp = LinearInterpolation(segments, noisydata[:, i, j])
            itp_data[:, i, j] = itp.(itp_x)
        end
    end

    itp_inf_l = zeros(length(itp_x), length(sensor_locs))
    for i = 1:length(sensor_locs)
        itp = LinearInterpolation(segments, unit_inf_line[:, i])
        itp_inf_l[:, i] = itp.(itp_x)
    end


    noisydata = itp_data
    unit_inf_line = itp_inf_l =#

    return data, noisydata, unit_inf_line, func_inf_lines
end

# Create BNN structure
num_neurons = 10
BNN = FastChain(
    FastDense(2, num_neurons, tanh),
    FastDense(num_neurons, num_neurons, tanh),
    FastDense(num_neurons, 1, tanh),
)

# Regularization, parameter variance and BNN number of params
num_params = length(initial_params(BNN))
alpha = 0.09
sig = sqrt(1.0 / alpha)

function bnn(x::AbstractFloat, z::AbstractFloat, x_middle, p)
    # x is normalised
    if x < x_middle
        y = BNN([x, z], p)[1]
    else #symmetry around L/2
        y = BNN([x_middle - (x - x_middle), z], p)[1]
    end
    return y
end

@model function bayesian_sciml(observations, x, z, sensor, num_params, inf_l, ::Type{T} = Float64) where {T}

    #prior parameters
    θ ~ MvNormal(zeros(num_params), sig .* ones(num_params))
    std ~ InverseGamma(2, 1)/100

    #Loop over the chosen sensors and load lanes
    preds = Array{T}(undef, length(x), length(z), length(sensor))

    pred = bnn.(x, z', last(sensor_locs) / 2, [θ])
    for i = 1:length(z)
        for j = 1:length(sensor)
            # call neural network prediction * influence line
            preds[:, i, j] = broadcast(*, inf_l[:, j], pred[:, i])
        end
    end
    #Flatten pred
    preds_flat = preds[:]

    #Likelihood
    observations ~ MvNormal(preds_flat, std)

end
