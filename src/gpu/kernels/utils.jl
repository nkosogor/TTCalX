# GPU Kernel Utilities

using CUDA

#==============================================================================#
#                        Constants for Speed of Light                          #
#==============================================================================#

const GPU_C = 2.99792e+8  # Speed of light in m/s


#==============================================================================#
#                     Inline Jones Matrix Operations                           #
#==============================================================================#

# These are device-side operations for use within CUDA kernels
# They operate on individual complex numbers, not arrays

"""
Multiply two 2x2 Jones matrices: C = A * B
Inputs are the 4 components of each matrix.
Returns tuple (Cxx, Cxy, Cyx, Cyy)
"""
@inline function jones_multiply(
    axx::ComplexF64, axy::ComplexF64, ayx::ComplexF64, ayy::ComplexF64,
    bxx::ComplexF64, bxy::ComplexF64, byx::ComplexF64, byy::ComplexF64
)
    cxx = axx*bxx + axy*byx
    cxy = axx*bxy + axy*byy
    cyx = ayx*bxx + ayy*byx
    cyy = ayx*bxy + ayy*byy
    return (cxx, cxy, cyx, cyy)
end

"""
Multiply Jones matrix by conjugate transpose: C = A * B'
"""
@inline function jones_multiply_conjtrans(
    axx::ComplexF64, axy::ComplexF64, ayx::ComplexF64, ayy::ComplexF64,
    bxx::ComplexF64, bxy::ComplexF64, byx::ComplexF64, byy::ComplexF64
)
    # B' = [conj(bxx) conj(byx); conj(bxy) conj(byy)]
    cxx = axx*conj(bxx) + axy*conj(bxy)
    cxy = axx*conj(byx) + axy*conj(byy)
    cyx = ayx*conj(bxx) + ayy*conj(bxy)
    cyy = ayx*conj(byx) + ayy*conj(byy)
    return (cxx, cxy, cyx, cyy)
end

"""
Conjugate transpose of a Jones matrix
"""
@inline function jones_conjtrans(
    xx::ComplexF64, xy::ComplexF64, yx::ComplexF64, yy::ComplexF64
)
    return (conj(xx), conj(yx), conj(xy), conj(yy))
end

"""
Determinant of a 2x2 Jones matrix
"""
@inline function jones_det(
    xx::ComplexF64, xy::ComplexF64, yx::ComplexF64, yy::ComplexF64
)
    return xx*yy - xy*yx
end

"""
Inverse of a 2x2 Jones matrix
Returns tuple (ixx, ixy, iyx, iyy)
"""
@inline function jones_inv(
    xx::ComplexF64, xy::ComplexF64, yx::ComplexF64, yy::ComplexF64
)
    d = jones_det(xx, xy, yx, yy)
    if abs(d) < eps(Float64)
        return (ComplexF64(1), ComplexF64(0), ComplexF64(0), ComplexF64(1))
    end
    inv_d = 1.0 / d
    return (yy*inv_d, -xy*inv_d, -yx*inv_d, xx*inv_d)
end

"""
Add two Jones matrices: C = A + B
"""
@inline function jones_add(
    axx::ComplexF64, axy::ComplexF64, ayx::ComplexF64, ayy::ComplexF64,
    bxx::ComplexF64, bxy::ComplexF64, byx::ComplexF64, byy::ComplexF64
)
    return (axx+bxx, axy+bxy, ayx+byx, ayy+byy)
end

"""
Frobenius norm of a Jones matrix
"""
@inline function jones_norm(
    xx::ComplexF64, xy::ComplexF64, yx::ComplexF64, yy::ComplexF64
)
    return sqrt(abs2(xx) + abs2(xy) + abs2(yx) + abs2(yy))
end


#==============================================================================#
#                    Diagonal Jones Matrix Operations                          #
#==============================================================================#

"""
Multiply diagonal Jones by full Jones: C = diag(a) * B
"""
@inline function diag_jones_multiply(
    axx::ComplexF64, ayy::ComplexF64,
    bxx::ComplexF64, bxy::ComplexF64, byx::ComplexF64, byy::ComplexF64
)
    return (axx*bxx, axx*bxy, ayy*byx, ayy*byy)
end

"""
Multiply full Jones by diagonal conjugate transpose: C = A * diag(b)'
"""
@inline function jones_multiply_diag_conjtrans(
    axx::ComplexF64, axy::ComplexF64, ayx::ComplexF64, ayy::ComplexF64,
    bxx::ComplexF64, byy::ComplexF64
)
    return (axx*conj(bxx), axy*conj(byy), ayx*conj(bxx), ayy*conj(byy))
end

"""
Inverse of diagonal Jones matrix
"""
@inline function diag_jones_inv(xx::ComplexF64, yy::ComplexF64)
    return (1.0/xx, 1.0/yy)
end

"""
Inner multiply for diagonal Jones matrices (used in stefcal)
Returns DiagonalJonesMatrix: (xx'*yxx + yx'*yyx, xy'*yxy + yy'*yyy)
But for diagonal, this simplifies to element-wise products
"""
@inline function diag_inner_multiply(
    gxx::ComplexF64, gyy::ComplexF64,  # G = input[i]
    mxx::ComplexF64, mxy::ComplexF64, myx::ComplexF64, myy::ComplexF64  # M = model[i,j]
)
    # GM = diag(g) * M
    gmxx = gxx * mxx
    gmxy = gxx * mxy
    gmyx = gyy * myx
    gmyy = gyy * myy
    
    # For diagonal output: inner_multiply(GM, V) for diagonal jones
    # Returns DiagonalJonesMatrix(GM.xx'*V.xx + GM.yx'*V.yx, GM.xy'*V.xy + GM.yy'*V.yy)
    # But this is called with V for numerator and GM for denominator
    
    return (gmxx, gmxy, gmyx, gmyy)
end


#==============================================================================#
#                          Geometric Calculations                              #
#==============================================================================#

"""
Compute geometric delay for one antenna
"""
@inline function geometric_delay(
    ant_x::Float64, ant_y::Float64, ant_z::Float64,
    l::Float64, m::Float64, n::Float64
)
    return (ant_x*l + ant_y*m + ant_z*n) / GPU_C
end

"""
Convert delay to fringe (complex exponential)
"""
@inline function delay_to_fringe(delay::Float64, frequency::Float64)
    ϕ = 2π * frequency * delay * 1im
    return exp(ϕ)
end


#==============================================================================#
#                            Thread Indexing                                   #
#==============================================================================#

"""
Get linear thread index for 1D kernel
"""
@inline function thread_index_1d()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
end

"""
Get 2D thread indices
"""
@inline function thread_index_2d()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return (i, j)
end

"""
Get 3D thread indices  
"""
@inline function thread_index_3d()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    return (i, j, k)
end


#==============================================================================#
#                        Kernel Launch Utilities                               #
#==============================================================================#

"""
Calculate optimal block and grid dimensions for 1D kernel
"""
function kernel_config_1d(N::Int; block_size::Int=256)
    blocks = cld(N, block_size)
    return (blocks,), (block_size,)
end

"""
Calculate optimal block and grid dimensions for 2D kernel
"""
function kernel_config_2d(N1::Int, N2::Int; block_size::Tuple{Int,Int}=(16, 16))
    blocks = (cld(N1, block_size[1]), cld(N2, block_size[2]))
    return blocks, block_size
end

"""
Calculate optimal block and grid dimensions for 3D kernel
"""
function kernel_config_3d(N1::Int, N2::Int, N3::Int; block_size::Tuple{Int,Int,Int}=(8, 8, 8))
    blocks = (cld(N1, block_size[1]), cld(N2, block_size[2]), cld(N3, block_size[3]))
    return blocks, block_size
end
