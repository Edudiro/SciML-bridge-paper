module FieldTest

export Truck, TruckName

@enum TruckName LeftTruck RightTruck

"""Truck used in the field test of the IJssel bridge."""
struct Truck
    "Truck name, either Truck1 or Truck2."
    name::TruckName
    "Transverse position of the center of the truck. [m]"
    center_tvs_pos::Real
    "Force per wheel, columns: left and right wheels; rows: axles. [kN]"
    wheel_forces::AbstractArray{T,2} where {T <: Real}
    "Longitudinal coordinate of each element of `wheel_forces`. The first axle is at 0. [m]"
    wheel_long_pos::AbstractArray{T,2} where {T <: Real}
    "Longitudinal coordinates of the elements of `wheel_forces`. [m]"
    wheel_tvs_pos::AbstractArray{T,2} where {T <: Real}

    """
        Truck(name::TruckName, center_tvs_pos::Real)

    Define `Truck` with its `name` and center's transverse positions (`center_tvs_pos`).
    """
    Truck(name::TruckName, center_tvs_pos::Real) = begin
        if name == RightTruck
            wheel_forces = [
                57.518 / 2  57.518 / 2;
                105.45 / 2  105.45 / 2;
                105.45 / 2  105.45 / 2;
                105.45 / 2  105.45 / 2;
                105.45 / 2  105.45 / 2;
            ]
            axle_dist = [1.94, 2.09, 1.345, 1.25]
            wheel_tvs_dist = 2.1500
        elseif name == LeftTruck
            wheel_forces = [
                58.86 / 2   58.86 / 2;
                107.91 / 2  107.91 / 2;
                107.91 / 2  107.91 / 2;
                107.91 / 2  107.91 / 2;
                107.91 / 2  107.91 / 2;
            ]
            axle_dist = [2.0, 1.82, 1.82, 1.82]
            wheel_tvs_dist = 2.1500
        end

        num_axle = size(wheel_forces, 1)
        wheel_long_pos = repeat([0, -cumsum(axle_dist)...], 1, 2)
        wheel_tvs_pos = hcat(
            (center_tvs_pos - wheel_tvs_dist / 2) * ones(num_axle),
            (center_tvs_pos + wheel_tvs_dist / 2) * ones(num_axle),
       )
        return new(name, center_tvs_pos, wheel_forces, wheel_long_pos, wheel_tvs_pos)
    end
end

end
