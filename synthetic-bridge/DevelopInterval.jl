function nonzerocount(V::Vector{Float64})
    count = 0
    for i = 1 : length(V)
        if V[i]!= 0.0
            count += 1
        end
    end
    return count
end

function nonzeros(V::Vector{Float64}, len::Int)
    result = zeros(len)
    count = 1
    for i = 1 : length(V)
        if V[i]!= 0.0
            result[count] = V[i]
            count += 1
        end
    end
    return result
end

function inner_value_fill(inner_values::Vector{Float64}, div::Int)
    step = Float64(1/div)
    fill = nonzerocount(inner_values)
    free = length(inner_values) - fill
    inner_values_sorted = sort(nonzeros(inner_values, fill))
    for i = 1 : fill
        if i == 1
            inner_values[fill + i] = inner_values_sorted[1] - step
            free -= 1
            if free == 0
                break
            end
        end
        inner_values[fill + i + 1] = inner_values_sorted[i] + step
        free -= 1
        if free == 0
            break
        end
    end

    return inner_values
end


function get_ll_divisions(width::Float64, num::Int)
    divisions = zeros(num)
    inner_values = zeros(num-2)

    if length(divisions) > 1
        divisions[2] = 1
    end

    if length(divisions) > 2

        div = 4
        inner_values[1] = 0.5
        while nonzerocount(inner_values) < length(inner_values)

            inner_values = inner_value_fill(inner_values, div)
            div *= 2

        end

        divisions[3:end] = inner_values[:]
    end

    return divisions * width
end
