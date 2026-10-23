# Needs python-casacore (as CI provides) and the example MS in examples/.
const EXAMPLE_MS = joinpath(dirname(@__DIR__), "examples", "20240524_090003_73MHz_ch0to5.ms")

casacore_available() = try
    init_pycasacore()
catch
    false
end

if casacore_available() && isdir(EXAMPLE_MS)
    @testset "MS I/O" begin
        @testset "read_ms_fast equals read_ms_to_gpu" begin
            for gpu in (false, is_gpu_available() ? true : false) |> unique
                a = read_ms_to_gpu(EXAMPLE_MS; gpu=gpu)
                b = read_ms_fast(EXAMPLE_MS; gpu=gpu)
                same(x, y) = typeof(x) == typeof(y) && Array(x) == Array(y)
                for f in fieldnames(GPUVisibilities)
                    @test same(getfield(a[1], f), getfield(b[1], f))
                end
                for f in fieldnames(GPUCalibration)
                    x, y = getfield(a[2], f), getfield(b[2], f)
                    @test x isa AbstractArray ? same(x, y) : x == y
                end
                for f in (:antenna_positions, :baselines, :channels, :phase_center, :uvw)
                    @test same(getfield(a[3], f), getfield(b[3], f))
                end
                @test a[3].phase_center_ra == b[3].phase_center_ra && a[3].phase_center_dec == b[3].phase_center_dec
                @test a[4] == b[4] && a[5] == b[5]
            end
        end

        @testset "write_ms_fast! writes what write_gpu_to_ms! writes" begin
            dir = mktempdir()
            ms_a, ms_b = joinpath(dir, "a.ms"), joinpath(dir, "b.ms")
            cp(EXAMPLE_MS, ms_a); cp(EXAMPLE_MS, ms_b)
            vis, _, _, bl, nrows = read_ms_to_gpu(ms_a; gpu=false)
            vis.xx .*= 0.5; vis.xy .+= 1im
            vis.flags[1:7:end, 2] .= true
            write_gpu_to_ms!(ms_a, vis, bl, nrows)
            write_ms_fast!(ms_b, vis, bl, nrows)
            ta = TTCalX.tables.table(ms_a); tb = TTCalX.tables.table(ms_b)
            for col in ("DATA", "FLAG", "UVW", "ANTENNA1", "ANTENNA2")
                @test ta.getcol(col) == tb.getcol(col)
            end
            ta.close(); tb.close()

            # A baseline map without the highest antenna: both writers leave its rows untouched.
            ms_c, ms_d = joinpath(dir, "c.ms"), joinpath(dir, "d.ms")
            cp(EXAMPLE_MS, ms_c); cp(EXAMPLE_MS, ms_d)
            top = maximum(maximum(k) for k in keys(bl))
            partial = Dict(k => v for (k, v) in bl if maximum(k) < top)
            @test length(partial) < length(bl)
            write_gpu_to_ms!(ms_c, vis, partial, nrows)
            write_ms_fast!(ms_d, vis, partial, nrows)
            tc = TTCalX.tables.table(ms_c); td = TTCalX.tables.table(ms_d)
            @test tc.getcol("DATA") == td.getcol("DATA") && tc.getcol("FLAG") == td.getcol("FLAG")
            tc.close(); td.close()
        end

        @testset "ms_mjd" begin
            @test 60454.37 < ms_mjd(EXAMPLE_MS) < 60454.38     # 2024-05-24 09:00 UTC
        end
    end
else
    @info "Skipping MS I/O tests (python-casacore or the example MS not available)"
end
