using MLDatasets
using Random, Distributions, LinearAlgebra, DataFrames, Statistics
using Images
using Flux.Losses

data, label = MLDatasets.MNIST.traindata()

# 
    train_num = 10
    test_num = 500
# 
x = reshape(data, (28, 28, :))
# 
    t = 28 * 28
    n = (train_num * t)
# 


function to_pix(x)
    return RGB(x, x ,x)
end

function draw_matrix(matrix)
    x = reshape(matrix, (28, 28))
    # a = x[:, :, 1]
    pixels = to_pix.(x)
end

function init_param()
    w1 = rand(Float64, (28, 28, train_num)) .- 0.5
    b1 = rand(Float64, (1, train_num)) .- 0.5

    w2 = rand(Float64, (10, 10)) .- 0.5
    b2 = rand(Float64, (1, 10)) .- 0.5
    return w1, b1, w2, b2
end

function ReLu(x)
    if x > 0
        return x
    else
        return 0
    end
end

function soft_max(Z)
    exp.(x) ./ sum(exp.(x))
end


w1, b1, w2, b2 = init_param()

train_x = x[1:n]    
train_x = reshape(train_x, (28, 28, :))


# Z1
z = [zeros(28,28) for i ∈ 1:10]
for i ∈ 1:10 
    xᵢ = train_x[:,:,i]
    wᵢ = w1[:,:,i]
    bᵢ = b1[:,i]
    z[i] = xᵢ * wᵢ .+ bᵢ
end

a1 = [zeros(28,28) for i ∈ 1:10]
for i ∈ 1:10
    zᵢ = z[i]
    a1[i] = ReLu.(zᵢ)
end
t2 = draw_matrix(a1[1])

# Using Cross-Entropy to get the loss
# the sum of logarithms
function cross_entropy(y_hat, y)
    return sum( q[i] * log(p[i]) for i ∈ 1:length(p))
end

y = [0, 2]
y_hat = [[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]

cross_entropy(y_hat, y)