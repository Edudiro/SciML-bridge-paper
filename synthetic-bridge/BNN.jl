#=
This file recreates a 2D bridge using
FE method to generate the ground truth
influence lines of a load at z,
measured by a sensor at (x,z = 0).

We will add random error to the ground truth
data and model it accounting for uncertainty.
A BNN will be used to predict the deflection
for any position in (x,z), resulting
in a predicted influence line
(constant z) for a given sensor.

=#

#packages and functions

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
using FLoops
using Random
using Distributions
using Turing
include("CALFEMhelpers.jl")
include("DevelopInterval.jl")
include("TrainingFunctionsDataDriven.jl")
include("TrainingFunctionsSciML.jl")

Turing.setadbackend(:tracker)

##Build bridge model
#= return data matrix containing
influence line from a set of load lanes
at any chosen sensor or set of sensors.
=#


# Create BNN structure
PURE_BNN= FastChain(
                FastDense(3,10,tanh),
                FastDense(10,10,tanh),
                FastDense(10,1),
)

# Regularization, parameter variance and BNN number of params
num_param = length(initial_params(PURE_BNN))
alpha = 0.09
sig = sqrt(1.0 / alpha)

#helper function

function pure_bnn(x::AbstractFloat, z::AbstractFloat, sensor::AbstractFloat, p)
    y = PURE_BNN([x, z, sensor], p)[1]
    return y
end


@model function bayesian_dd(observations, x, z, sensor, ::Type{T} = Float64) where {T}

    #prior parameters
    θ ~ MvNormal(zeros(num_param), sig .* ones(num_param))
    std ~ InverseGamma(2, 1)/100

    #Loop over the chosen sensors and load lanes
    preds = Array{T}(undef, length(x), length(z), length(sensor))
    for i = 1:length(z)
        for j = 1:length(sensor)
            # call neural network prediction
            preds[:, i, j] = pure_bnn.(x, z[i], sensor[j], [θ])
        end
    end

    #Flatten pred
    preds_flat = preds[:]

    #Likelihood
    observations ~ MvNormal(preds_flat, std)

end
