





# ------------------------------------------------------
# BUILD SIMPLE MODEL AND SUBDIVIDE
# ------------------------------------------------------
function trainsystemSciML(load_lane_offsets::AbstractVector, sensor_locs::Vector, noisydata::Array, zdim::Int)

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
        unit_inf_line[:,i] =2 .* translation_at_sensor.(sensor_id[i], P, 1:2:length(segments)*2)
    end


    # Create unit influence line for validation (refactor this later in a good way.)
    all_unit_inf_lines = zeros(length(segments), length(segments))
    all_max_inf_lines = zeros(length(segments), length(segments))
    for i=1:length(segments)
        _unit_infline =2 .*  translation_at_sensor.(1+2*(i-1), P, 1:2:length(segments)*2)
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


    meas_inf_line = noisydata

    # ------------------------------------------------------
    # DEFINE NEURAL NETWORK
    # ------------------------------------------------------

    # The neural network NN(x,y) is defined and the weights initialized.
    num_neurons = 10
    NN = FastChain(
        FastDense(2, num_neurons, tanh),
        FastDense(num_neurons, num_neurons, tanh),
        FastDense(num_neurons, 1, tanh),
    )
    pinit = initial_params(NN)

    # ------------------------------------------------------
    # DEFINE LOSS FUNCTION
    # ------------------------------------------------------
    # Define callback to help visualizing the results
    # Turn off for optimizing

    global iter = 0


    function loss(p)

        #=
        # Helper function
        function find_idx(x)
            findfirst(isequal(x), segments)
        end
            =#

        function tlv_nn(x::AbstractFloat, z::AbstractFloat, x_middle::AbstractFloat)
            if x < x_middle
              y = NN([x, z], p)[1]
            else
              y = NN([x_middle - (x - x_middle),z], p)[1]
            end
            #y = NN([x_middle - (x - x_middle),z], p)[1]
            return y
        end

        function dfdx(x, z, mid)
            epsilon=cbrt(eps(Float32))
            return (tlv_nn(x + epsilon, z, mid) - tlv_nn(x - epsilon, z, mid)) / (2 * epsilon)
        end

        function dfdx(x, z, mid)
            epsilon=cbrt(eps(Float32))
            return (tlv_nn(x + epsilon, z, mid) - tlv_nn(x - epsilon, z, mid)) / (2 * epsilon)
        end

        function get_mse_loss(f_pred::AbstractArray, uy_il::AbstractArray, meas_il::AbstractArray)
            mse_loss = 0

            for i=1:size(f_pred)[2] #For each load case
                for j=1:size(uy_il)[2] # For each sensor
                uy_il_pred = broadcast(*, uy_il[:, j], f_pred[:, i])
                mse_loss += mean(abs2.(uy_il_pred .- meas_il[:,i,j])) / (size(f_pred)[2] + size(uy_il)[2])
                end
            end

            return mse_loss
        end

        function get_ddx_loss(zr, x_middle)
            ddx_loss = 0.0

            for z in collect(zr)
                ddx_loss += abs(dfdx(x_middle - eps(Float16), z, x_middle) - dfdx(x_middle + eps(Float16), z, x_middle))
            end

            return ddx_loss
        end
            #=

        function get_mono_loss(f_pred::AbstractArray)
            mono_loss = 0

            for i=1:size(f_pred)[1] #For each segment
                mono_loss += sum_derivatives(f_pred[i, :])
            end

            return mono_loss
        end

        function sum_derivatives(u::Vector)
            du = 0
            _du = 0
            for i = 1 : length(u)
                if i == 1
                    _du = u[i + 1] - u[i]

                elseif i == length(u)
                    _du = u[i] - u[i - 1]

                else
                    _du = (u[i + 1] - u[i - 1]) / 2
                end

                if _du > 0
                    du += _du
                end
            end

            return du
        end
            =#


        # Calculate predicted tlv values at measured positions
        f_tlv_pred_load = tlv_nn.(segments, load_lane_offsets', x_middle)

        # Calculate predicted tlv values at all positions (Optimize later)
        #f_tlv_pred_all = tlv_nn.(xr, zr', x_middle)

        # Calculate MSE loss
        mse_loss = get_mse_loss(f_tlv_pred_load, unit_inf_line, meas_inf_line)

        # Midpoint derivative loss
        ddx_loss = get_ddx_loss(load_lane_offsets, x_middle)
        # Enforce monotonicity loss function
       #mono_loss = get_mono_loss(f_tlv_pred_all)

        #=
        # Calculate BC loss
        bc_idx=broadcast(find_idx,support_pos)
        bc_pred=f_tlv_pred_load[bc_idx,1]
        bc_loss = mean(abs2.(bc_pred .- 1.0))
        =#


        #=
         # Not in Zygote
        Zygote.ignore() do
            #global LossMSE
            #global count
            LossMSE[count] = mse_loss
            #LossBC[count] = bc_loss
            #LossMono[count] = mono_loss
            count += 1
        end
        =#


        # return loss value
        return 1 * mse_loss + 0.0001 * ddx_loss#+ 1 * mono_loss#+ =#1 * bc_loss
    end

    # ------------------------------------------------------
    # TRAIN NEURAL NETWORK
    # ------------------------------------------------------
    track_loss = []
    NN_callback = function (p, l)
        iter += 1
        if iter % 100 == 1
            #@show l
            append!(track_loss,l)
            #= function tlv_nn(x::AbstractFloat, z::AbstractFloat, x_middle::AbstractFloat)
                if x < x_middle
                  y = NN([x, z], p)[1]
                else
                  y = NN([x_middle - (x - x_middle),z], p)[1]
                end
                #y = NN([x_middle - (x - x_middle),z], p)[1]
                return y
            end


            tmp_pred = broadcast(*, unit_inf_line[:,1],tlv_nn.(segments, load_lane_offsets', x_middle)[:,1])

            plt = plot(segments, tmp_pred)
            plot!(plt, segments, meas_inf_line[:,1,1])

            display(plt) =#
        end

        false
    end

    iters = 5000

    #=count = 1
    LossMSE = zeros(Float64,iters*2)
    LossBC = zeros(Float64,iters*2)
    LossMono = zeros(Float64,iters*2)
    =#

    res1 = DiffEqFlux.sciml_train(loss, pinit,
    ADAM(0.05), maxiters=iters, save_best=true, cb = NN_callback)

    # refine with a lower learning rate
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer,
        ADAM(0.01), maxiters=iters, save_best=true, cb = NN_callback)

    p = res2.minimizer

    #@show(p)
    return p,track_loss
end





function getinfSciML(NN, p, all_uy_il, xr, zr)

    x_middle = last(xr) / 2
    # Helper function
    function tlv_nn(x, z)
        if x < x_middle
          y = NN([x, z], p)[1]
        else
          y = NN([x_middle - (x - x_middle), z], p)[1]
        end

        #y = NN([x, z], p)[1]
        return y
    end

    num_sensors = size(all_uy_il, 2)
    realinfs = zeros(length(xr), length(zr), num_sensors)

    for i = 1:num_sensors

        #Get unit influence line
        uy_il = all_uy_il[:, i]

        # Discretize tlv function
        X = xr' .* ones(length(zr))
        Z = ones(length(xr))' .* zr
        ff = tlv_nn.(X, Z)

         # Create influence surfaces
        infsrf = repeat(uy_il, 1, length(zr))'.*ff
        realinfs[:, :, i] = infsrf'
    end
    return realinfs
end
