#==============================================================================#
#             Reusable per-snapshot buffers for zest_sky_rfi!                  #
#==============================================================================#
#
# A zest of one OVRO-LWA snapshot (352 antennas, 48 channels) allocates several
# GB of device memory: a model per direction (190 MB), a square model per
# direction (190 MB in :mixed), the square data, the restore basis, and ~100
# visibility-sized temporaries (residual powers, restore normal equations).  A
# long-lived process that peels back to back with a CUDA.jl soft memory limit
# then collects garbage inside the solve.  `PeelWorkspace` keeps every one of
# these buffers and hands them out again on the next call with the same shapes,
# so a warm zest allocates almost nothing on the device.
#
# The buffers are filled exactly as the allocating code fills fresh arrays
# (zeros where it starts from zeros), and every computation is the same
# broadcast or reduction into a reused destination, so results are bitwise
# identical with and without a workspace.

"""
    PeelWorkspace()

Device (or host) buffers that `zest_sky_rfi!(...; workspace=ws)` reuses from one
call to the next.  Buffers are created on first use and kept, keyed on their role,
element type, device and shape; when a call has another (Nant, Nbase, Nfreq,
precision, device) than the previous one, all buffers are freed first.  More
directions than before only add buffers.

Not for concurrent calls.  Nothing the zest returns refers to the workspace (the
returned gains are fresh arrays).  `workspace_bytes(ws)`, `ws.nalloc` (buffers
created so far) and `ws.nreset` (shape changes) describe it; `free_workspace!(ws)`
releases its memory.
"""
mutable struct PeelWorkspace
    key::Any
    bufs::Dict{Any,Vector{Any}}
    nalloc::Int
    nreset::Int
end
PeelWorkspace() = PeelWorkspace(nothing, Dict{Any,Vector{Any}}(), 0, 0)

function Base.show(io::IO, ws::PeelWorkspace)
    print(io, "PeelWorkspace(key=", ws.key, ", buffers=", sum(length, values(ws.bufs); init=0),
          ", ", round(workspace_bytes(ws) / 2^20; digits=1), " MiB, nalloc=", ws.nalloc,
          ", nreset=", ws.nreset, ")")
end

"""Bytes held by the workspace's buffers."""
workspace_bytes(ws::PeelWorkspace) = sum((sizeof(a) for pool in values(ws.bufs) for a in pool); init=0)

"""Free every buffer of `ws` (device memory goes back to the pool at once)."""
function free_workspace!(ws::PeelWorkspace)
    for pool in values(ws.bufs), a in pool
        a isa CuArray && CUDA.unsafe_free!(a)
    end
    empty!(ws.bufs)
    ws.key = nothing
    ws
end

"""Start a call with shape `key`: drop the buffers of another shape."""
function _ws_prepare!(ws::PeelWorkspace, key)
    if ws.key != key
        if ws.key !== nothing
            free_workspace!(ws)
            ws.nreset += 1
        end
        ws.key = key
    end
    ws
end

"""
Buffer number `i` of role `tag`, element type `T`, shape `dims` on the device (`gpu`)
or host; zero-filled unless `zero=false` (the caller overwrites all of it).
"""
function _ws_buf!(ws::PeelWorkspace, tag, ::Type{T}, dims::Dims, gpu::Bool, i::Int=1;
                  zero::Bool=true) where {T}
    pool = get!(Vector{Any}, ws.bufs, (tag, T, gpu, dims))
    while length(pool) < i
        push!(pool, gpu ? CuArray{T}(undef, dims) : Array{T}(undef, dims))
        ws.nalloc += 1
    end
    a = pool[i]
    zero && fill!(a, Base.zero(T))
    a
end

"""Visibilities number `i` of role `tag` (as `GPUVisibilities(Nb, Nf)` when zeroed)."""
function _ws_vis!(ws::PeelWorkspace, tag, Nb::Int, Nf::Int, gpu::Bool, i::Int=1; zero::Bool=true)
    c(q) = _ws_buf!(ws, (tag, q), ComplexF64, (Nb, Nf), gpu, i; zero=zero)
    GPUVisibilities(c(1), c(2), c(3), c(4), _ws_buf!(ws, (tag, 5), Bool, (Nb, Nf), gpu, i; zero=zero))
end

"""Square visibilities number `i` of role `tag` (as `GPUSquareVisibilities(Na, Nf; T)`)."""
function _ws_square!(ws::PeelWorkspace, tag, Na::Int, Nf::Int, ::Type{T}, gpu::Bool, i::Int=1) where {T}
    c(q) = _ws_buf!(ws, (tag, q), T, (Na, Na, Nf), gpu, i)
    GPUSquareVisibilities(c(1), c(2), c(3), c(4))
end

"""`deepcopy_gpu(vis)` into the workspace's buffers of role `tag`."""
function _ws_copy!(ws::PeelWorkspace, tag, vis::GPUVisibilities)
    c = _ws_vis!(ws, tag, size(vis.xx)..., _is_gpu(vis.xx); zero=false)
    for f in (:xx, :xy, :yx, :yy, :flags)
        copyto!(getfield(c, f), getfield(vis, f))
    end
    c
end
