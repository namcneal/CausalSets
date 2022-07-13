using ForwardDiff
using ForwardDiff: jacobian

using Einsum
using LinearAlgebra
using TensorCast

Input = Union{Int64,
              Float32, 
              Float64,
}

VectorInput = Union{Vector{Int64}, 
                    Vector{Float32},          
                    Vector{Float64},
                    Vector
}

function sum2(arr::Array)
    return sum(arr.^2)
end

##
function schwarzschild(point::VectorInput;
                       c::Float64=1.0, G::Float64=1.0, M::Float64=0.5)

    rs = 2 * G * M / c^2

    return diagm([-(1 - rs / point[2]), 
                 1/(1 - rs / point[2]),
                 point[2]^2,
                 point[2]^2 * sin(point[3])^2
                ])
end

function analytic_metric_jacobian(point::VectorInput;
                                  c::Float64=1.0, 
                                  G::Float64=1.0, 
                                  M::Float64=0.5)
    rs = 2 * G * M / c^2
    d = length(point)
    ∂g = zeros(Float64, (d,d,d))

    r = point[2]
    θ = point[3]
    ∂g[1, 1, 2] = - rs / r^2
    ∂g[2, 2, 2] = - rs / (rs - r)^2
    ∂g[3, 3, 2] = 2r
    ∂g[4, 4, 2] = 2r  * sin(θ)^2

    ∂g[4, 4, 3] = r^2 * 2*sin(θ)*cos(θ)
    return ∂g
end

function analytic_christoffel(point::VectorInput;
                              c::Float64=1.0, 
                              G::Float64=1.0, 
                              M::Float64=0.5)
    d = length(point)
    Γ = zeros(Float64, (d,d,d))

    # From Carroll p. 206. Going across each row:
    # Carroll sets c = 1, so I won't touch other cases for now
    @assert c ≈ 1

    r = point[2]
    θ = point[3]
    Γ[2,1,1] =  G*M / r^3 * (r - 2G*M)
    Γ[2,2,2] = -G*M / (r  * (r - 2G*M))
    Γ[1,1,2] =  G*M / (r  * (r - 2G*M))
    Γ[1,2,1] =  Γ[1,1,2]    # Symmetry


    Γ[3,2,3] = 1 / r
    Γ[3,3,2] = Γ[3,2,3]     # Symmetry
    Γ[2,3,3] = -(r - 2G*M)
    Γ[4,2,4] = 1 / r
    Γ[4,4,2] = Γ[4,2,4]     # Symmetry

    Γ[2,4,4] = -(r - 2G*M) * sin(θ)^2
    Γ[3,4,4] = -sin(θ)*cos(θ)
    Γ[4,3,4] = cot(θ)
    Γ[4,4,3] = Γ[4,3,4]     # Symmetry
    return Γ
end

function analytic_christoffel_jacobian(point::VectorInput; 
                                       c::Float64=1.0, 
                                       G::Float64=1.0, 
                                       M::Float64=0.5)                         
    d  = length(point)
    ∂Γ = zeros(Float64, (d,d,d,d))

    # Carroll sets c = 1, so I won't touch other cases for now
    @assert c ≈ 1

    # From Carroll p. 206. Going across each row, but differentiating
    # each term. Subdivided by row.
    # The last index is the derivative. The first three are the same 
    # as the Christoffel symbol, i.e. [top, left, right, ∂]
    r = point[2]
    θ = point[3]
    
    ∂Γ[2,1,1,2] = 2G*M * (3G*M - r) / r^4
    ∂Γ[2,2,2,2] = -2G*M * (G*M - r) / (r^2 * (r - 2G*M)^2)
    ∂Γ[1,1,2,2] =  2G*M * (G*M - r) / (r^2 * (r - 2G*M)^2)
    ∂Γ[1,2,1,2] =  ∂Γ[1,1,2,2] # Symmetry

    ∂Γ[3,2,3,2] = -1 / r^2
    ∂Γ[3,3,2,2] = ∂Γ[3,2,3,2]  # Symmetry
    ∂Γ[2,3,3,2] = -1
    ∂Γ[4,2,4,2] = -1 / r^2
    ∂Γ[4,4,2,2] = ∂Γ[4,2,4,2]  # Symmetry

    ∂Γ[2,4,4,2] = -1 * sin(θ)^2
    ∂Γ[2,4,4,3] = -(r - 2G*M) * 2 * sin(θ) * cos(θ)
    ∂Γ[3,4,4,3] = -cos(θ)*cos(θ) + sin(θ)*sin(θ)

    ∂Γ[4,3,4,3] = - csc(θ)^2
    ∂Γ[4,4,3,3] = ∂Γ[4,3,4,3] # Symmetry

    return ∂Γ
end

# point = [0, 2, π/4, 0]
# d = length(point)

##
function issymmetric(A::Matrix{Float64})
    should_be_zero = A .- A'
    mag = sum(should_be_zero.^2)
    return isapprox(mag, 0.0, rtol=1e-3)
end

function isasymmetric(A::Matrix{Float64})
    should_be_zero = A .+ A'
    mag = sum(should_be_zero.^2)

    return isapprox(mag, 0.0, rtol=1e-3)
end  

function test_symmetry_checks()
    mat  = rand(d,d)
    sym  = rand(d,d); sym  .+= sym'
    asym = rand(d,d); asym .-= asym' 

    ## Testing the symmetric check
    @assert  issymmetric(sym)
    @assert !issymmetric(mat)
    @assert !issymmetric(asym)

    # Testing the asymmetric check
    @assert  isasymmetric(asym)
    @assert !isasymmetric(mat)
    @assert !isasymmetric(sym)
end

# test_symmetry_checks()
##

function metric_jacobian(metric::Function, point::VectorInput)
    ∂g = jacobian(x->metric(x), point)

    d = length(point)
    reshape(∂g, (d,d,d))
end

function test_metric_jacobian_symmetry(∂g::Array{Float64}, point::VectorInput)
    d = length(point)

    # Checking to make sure each component on the derivative index is symmetric
    for i in 1:d
        @assert issymmetric(∂g[:,:,i])
    end
end
function test_metric_jacobian_symmetry(metric::Function, point::VectorInput)
    ∂g = metric_jacobian(metric, point)

    return test_metric_jacobian_symmetry(∂g, point)
end

function test_schwarzschild_metric_jacobian(point::VectorInput)
    analytic = analytic_metric_jacobian(point)
    computed = metric_jacobian(schwarzschild, point)
    return @assert sum((analytic .- computed).^2) ≈ 0
end

# test_schwarzschild_metric_jacobian(point)
# test_metric_jacobian_symmetry(schwarzschild, point)

##
function christoffel(metric::Function, point::VectorInput)
    d = length(point)
    
    # Inverse of the metric here
    g   = inv(metric(point))

    # Derivative of the (non-inverse) metric
    ∂g = metric_jacobian(metric, point)

    # Need to reshape on the forward due to how ForwardDiff computes the jacobian
    ∂g = reshape(∂g, (d,d,d))

    Γ = 0 .* ∂g
    for ρ=1:d, μ=1:d, ν=1:d
        for σ=1:d
            Γ[σ,μ,ν] += g[σ, ρ]/2 * (∂g[ν,ρ,μ] + ∂g[ρ,μ,ν] - ∂g[μ,ν,ρ])
        end
    end

    return Γ
end


function test_schwarzschild_christoffel(point::VectorInput)
    analytic = analytic_christoffel(point)
    computed = christoffel(schwarzschild, point)

    @assert sum((analytic .- computed).^2) ≈ 0
end

function test_christoffel_symmetry(Γ::Array{Float64}, point::VectorInput)
    # Check the symmetry of the lower two indices for each upper
    d = length(point)
    for up in 1:d
        @assert issymmetric(Γ[up,:,:])
    end
end
function test_christoffel_symmetry(metric::Function, point::VectorInput)
    # Check the symmetry of the lower two indices for each upper
    Γ = christoffel(metric, point)
    return test_christoffel_symmetry(Γ, point)
end


# test_schwarzschild_christoffel(point)
# test_christoffel_symmetry(schwarzschild, point)

##
function christoffel_jacobian(metric::Function, point::VectorInput)
    f = x->christoffel(metric, x)
    
    # Need to reshape due to how ForwardDiff computes the jacobian
    d = length(point)
    return reshape(jacobian(f, point), (d,d,d,d))
end

function test_schwarzschild_christoffel_jacobian(point::Vector)
    analytic = analytic_christoffel_jacobian(point)
    computed = christoffel_jacobian(schwarzschild, point)
    mse = (analytic .- computed).^2

    d = length(point)
    for i=1:d, j=1:d, k = 1:d, l=1:d
        @assert mse[i,j,k,l] ≈ 0 "Error at Christoffel index [$(i),$(j),$(k)] at derivative  $(l) ⟹  ∂Γ[$(i),$(j),$(k), $(l)]"
    end
end

function test_christoffel_jacobian_symmetry(∂Γ::Array{Float64}, point::VectorInput)
    d = length(point)

    # Check the symmetry of the lower indices for each upper and each derivative
    for up in 1:d
        for der in 1:d
            @assert issymmetric(∂Γ[up,:,:,der])
        end
    end
end
function test_christoffel_jacobian_symmetry(metric::Function, point::VectorInput)
    ∂Γ = christoffel_jacobian(metric, point)
    return test_christoffel_jacobian_symmetry(∂Γ, point)
end

# test_christoffel_jacobian_symmetry(schwarzschild, point)
# test_schwarzschild_christoffel_jacobian(point)

##
function riemannian(metric::Function, point::VectorInput)
    Γ  = christoffel(metric, point)
    ∂Γ = christoffel_jacobian(metric, point)

    # I'm assuming I'll have to reshape due to something about 
    # how the ForwardDiff tape 
    d  = length(point)
    Γ  = reshape(Γ, (d,d,d))
    ∂Γ = reshape(∂Γ, (d,d,d,d))

    # Testing to make sure they stayed correct for all the manipulation.
    # Just a basic test for symmetry, though.
    # test_christoffel_symmetry(Γ, point)
    # test_christoffel_jacobian_symmetry(∂Γ, point)

    R = 0 * ∂Γ
    for ρ=1:d, σ=1:d, μ=1:d, ν=1:d
        R[ρ,σ,μ,ν] = ∂Γ[ρ,ν,σ,μ] - ∂Γ[ρ,μ,σ,ν]

        for λ=1:d
            R[ρ,σ,μ,ν] += Γ[ρ,μ,λ]*Γ[λ,ν,σ] - Γ[ρ,ν,λ]*Γ[λ,μ,σ]
        end 
    end

    return R
end

function test_riemannian_symmetry(Riem::Array{Float64}, point::VectorInput)
    d = length(point)
    for up in 1:d
        for down in 1:d
            @assert isasymmetric(Riem[up,down,:,:]) "Failed asymmetry on indices [$(up), $(down), :,:]"
        end
    end
end
function test_riemannian_symmetry(metric::Function, point::VectorInput)
    Riem = riemannian(metric, point)
    return test_riemannian_symmetry(Riem, point)
end

# riemannian(schwarzschild, point)
# test_riemannian_symmetry(schwarzschild, point)

##

function ricci(metric::Function, point::VectorInput)
    Riem  = riemannian(metric, point)

    d = length(point)
    R = zeros(Float64, (d,d))

    # ForwardDiff changes the type of the tensor to something like
    # ForwardDiff.Dual{ForwardDiff.Tag{var"#130#131", Float32}, Float64, 12}
    if !(Riem[1,1,1,1] isa Float64)
        R = 0 .* Riem[:,:,1,1]
    end

    for μ=1:d, ν=1:d
        for λ=1:d
            R[μ,ν] += Riem[λ,μ,λ,ν]
        end
    end

    return R
end

function scalar(metric::Function, point::VectorInput)
    g_inv = inv(metric(point))
    R     = ricci(metric, point)

    S = 0
    for μ=1:d, ν=1:d
        S += g_inv[μ,ν] * R[μ,ν]
    end

    return S
end

# @assert sum2(ricci(schwarzschild,  point)) < 1e-7
# @assert scalar(schwarzschild,      point)  < 1e-7

##
function EFE_LHS(metric::Function, point::VectorInput)
    g = metric(point)
    R = ricci(metric,  point)
    S = scalar(metric, point)

    return @. R - S/2 * g
end

