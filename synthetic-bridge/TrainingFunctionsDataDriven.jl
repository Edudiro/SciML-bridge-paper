





# ------------------------------------------------------
# BUILD SIMPLE MODEL AND SUBDIVIDE
# ------------------------------------------------------
function trainsystemDD(load_lane_offsets::AbstractVector, sensor_locs::Vector, noisydata::Array, zdim::Int)
    # Element properties
    E = 200e9    # [Pa]
    Iy = 8.2e-3  # [m^4]
    P = 500.0e3  # [N]
    A = 2.0e-3   # [m^2]

    ep = [E A Iy]'

    # Bridge x-axis geometry
    elem_geo = [0.0, 20.0, 40.0, 60.0]
    L = last(elem_geo)
    step = steps # size of steps
    spans = [20.0, 20.0, 20.0]
    support_pos = [0, cumsum(spans)...]
    x_middle = support_pos[end]/2
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
    (K2, f2, newBc, NewEdof, segments) = rebuild_linear_system(L, step, elem_geo, EP, bc, Edof, f,2)

    # Create ranges in x and z direction
    xr = range(0, stop=L, length=length(segments))
    zr = range(0, stop=bridge_width, length=zdim)


    # ------------------------------------------------------
    # CREATE GROUND TRUTH
    # ------------------------------------------------------

    # Get index of current sensor
    sensor_id = (x -> (findfirst(isequal(x), segments) - 1) * 2 + 1).(sensor_locs)


    # Ground truth abiding to boundary condition
    function tlv_f(x, z)
        idx = support_pos[1:(end - 1)] .<= x .<= support_pos[2:end]
        span = spans[idx][1]
        cs_span = support_pos[1:(end - 1)][idx][1]
        fx = 1 - 0.5 * span / max_span * sin(π * (x - cs_span) / span)
        fz = 1 - sqrt(z / (2 * bridge_width))
        return fx * fz
    end

    # ------------------------------------------------------
    # COMPUTE VERTICAL TRANSLATION INFL. LINE FOR SENSOR
    # ------------------------------------------------------
    function translation_at_sensor(sensor_id, load, load_id)
        f=zeros(size(f2))
        f[load_id] = load
        (u, fb) = solveq(K2,f,newBc)
        return u[sensor_id]
    end


    # Initialize unit influence line
    unit_inf_line = zeros(length(segments), length(sensor_locs))

    # Fill unit influence line for training
    for i=1:length(sensor_locs)
        unit_inf_line[:,i] = translation_at_sensor.(sensor_id[i], P, 1:2:length(segments)*2)
    end

    # Create unit influence line for validation (refactor this later in a good way.)
    all_unit_inf_lines = zeros(length(segments), length(segments))
    all_max_inf_lines = zeros(length(segments), length(segments))
    for i=1:length(segments)
        _unit_infline = translation_at_sensor.(1+2*(i-1), P, 1:2:length(segments)*2)
        all_unit_inf_lines[:,i] = _unit_infline
        all_max_inf_lines[:,i] = _unit_infline .* tlv_f.(segments, zeros(length(segments)))
    end

    # ------------------------------------------------------
    # CREATE TRAINING DATA
    # ------------------------------------------------------

    # load carried by the girder
    func_inf_lines = tlv_f.(segments, load_lane_offsets')

    meas_inf_line = zeros(length(segments), length(load_lane_offsets), length(sensor_locs))

    for i=1:length(sensor_locs)
        meas_inf_line[:,:,i] = unit_inf_line[:, i] .* func_inf_lines
    end


    function addnoise(data, scaledata, scale, seed = 12345)
        Random.seed!(seed)
        d = Normal(0.0, scale * maximum(abs.(scaledata)))
        noisydata = data .+ rand(d, size(data))
        return noisydata
    end


    #meas_inf_line = addnoise(meas_inf_line, all_max_inf_lines, 0.05)

    #=
    xs=collect(range(0,8*pi,length=100))
    ys = sin.(xs)
    ys = addnoise(ys, 0.05)
    Plots.scatter(xs, ys)
    =#

    meas_inf_line = noisydata

    # ------------------------------------------------------
    # DEFINE NEURAL NETWORK
    # ------------------------------------------------------

    # The neural network NN(x,y) is defined and the weights initialized.
    num_neurons = 10
    NN = FastChain(
        FastDense(3, num_neurons, tanh),
        FastDense(num_neurons, num_neurons, tanh),
        FastDense(num_neurons, 1),
    )

    pinit = initial_params(NN)

    # ------------------------------------------------------
    # DEFINE LOSS FUNCTION
    # ------------------------------------------------------
    # Define callback to help visualizing the results
    # Turn off for optimizing

    function NN_callback(p, l)
        @show l
        false
    end


    function loss(p)

        # Helper function
        function tlv_nn(x::AbstractFloat, z::AbstractFloat, sensor::AbstractFloat)
            y = NN([x, z, sensor], p)[1]
          return y
        end
        #=
        function get_mse_loss(f_pred::AbstractArray, meas_il::AbstractArray)
            mse_loss = 0

            for i=1:size(f_pred)[2] #For each load case
                for j=1:size(meas_il)[3] # For each sensor
                mse_loss += mean(abs2.(f_pred[:, i] .- meas_il[:,i,j]))
                end
            end

            return mse_loss / (size(f_pred)[2] + size(meas_il)[3])
        end =#

        function get_mse_loss(tlv_nn, segments::Vector, load_lane_offsets::Vector,
                                        sensor_locs::Vector, meas_il::AbstractArray)
            mse_loss = 0

            # For all x positions
            for i = 1:length(segments)
                # For all y positions
                for j = 1:length(load_lane_offsets)
                    # For all sensor positions in x
                    for k = 1:length(sensor_locs)
                        mse_loss +=  abs2(tlv_nn(segments[i], load_lane_offsets[j], sensor_locs[k]) - meas_il[i, j, k])
                    end
                end
            end

            return mse_loss / (length(segments) + length(load_lane_offsets) + length(sensor_locs))
        end

        # Calculate predicted influence values at measured positions
        #f_tlv_pred_load = tlv_nn.(segments, load_lane_offsets')

        # Calculate MSE loss
        mse_loss = get_mse_loss(tlv_nn, segments, load_lane_offsets, sensor_locs, meas_inf_line)


        # return loss value
        return mse_loss
    end

    # ------------------------------------------------------
    # TRAIN NEURAL NETWORK
    # ------------------------------------------------------
    iters = 10000

    res1 = DiffEqFlux.sciml_train(loss, pinit,
        ADAM(0.05), maxiters=iters, save_best=true, cb=NN_callback)

    # refine with a lower learning rate
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer,
        ADAM(0.003), maxiters=iters, save_best=true, cb=NN_callback)

    p = res2.minimizer


    # ------------------------------------------------------
    # PLOT RESULTS
    # ------------------------------------------------------

    plotlyjs()
    #=
    # Prediction function
    function tlv_nn(x::AbstractFloat, z::AbstractFloat)
              y = NN([x, z], p)[1]
            return y
        end

    # Save data to arrays
    X = xr' .* ones(length(zr))
    Z = ones(length(xr))' .* zr
    ff = tlv_f.(X, Z)
    nn = tlv_nn.(X, Z)

    # Create influence surfaces
    trueinf = repeat(unit_inf_line[:,1], 1, length(zr))'.*ff
    predinf = nn =#


    # Collect out values
    truefunc = tlv_f
    predfunc = NN
    params = p
    il_out = all_unit_inf_lines
    data = meas_inf_line

    return params
end



    function getinf(tlv_f, all_uy_il, xr, zr)
        num_sensors = size(all_uy_il, 2)
        realinfs = zeros(length(xr), length(zr), num_sensors)

        for i = 1:num_sensors

            #Get unit influence line
            uy_il = all_uy_il[:, i]

            # Discretize tlv function
            X = xr' .* ones(length(zr))
            Z = ones(length(xr))' .* zr
            ff = tlv_f.(X, Z)

             # Create influence surfaces
            infsrf = repeat(uy_il, 1, length(zr))'.*ff
            realinfs[:, :, i] = infsrf'
        end
        return realinfs
    end


  function getinfDD(NN, p, all_uy_il, xr, zr)

    # Helper function
    function tlv_nn(x::AbstractFloat, z::AbstractFloat, sensor::AbstractFloat)
        y = NN([x, z, sensor], p)[1]
      return y
    end

    # Initialize return value
    num_sensors = size(all_uy_il, 2)
    predinfs = zeros(length(xr), length(zr), num_sensors)

    # For all x positions
    for i = 1:length(xr)
        # For all y positions
        for j = 1:length(zr)
            # For all sensor positions in x
            for k = 1:length(xr)
                predinfs[i, j, k] = tlv_nn(xr[i]/67.5, zr[j]/5.0, xr[k]/67.5) #* all_uy_il[i, k]
            end
        end
    end

    return predinfs
end
