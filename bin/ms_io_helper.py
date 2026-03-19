#!/usr/bin/env python3
"""
MS I/O helper for TTCalX native reader.
Reads/writes MS columns as raw binary via stdout/stdin pipes.

Usage:
  python3 ms_io_helper.py read  <ms_path> <column>
  python3 ms_io_helper.py write <ms_path> <column>

Binary protocol (for each array):
  int32   ndim
  int64[] shape (ndim elements)
  int32   dtype_code (0=complex128, 1=float64, 2=int32, 3=uint8)
  bytes   raw data (C-contiguous)

READ outputs 8 arrays to stdout:
  data, flags, ant1, ant2, uvw, chan_freq, positions, phase_dir

WRITE reads 2 arrays from stdin:
  data, flags
"""
import sys
import struct
import numpy as np


def write_array(out, arr):
    """Write self-describing binary array to stream."""
    arr = np.ascontiguousarray(arr)
    out.write(struct.pack('<i', arr.ndim))
    for s in arr.shape:
        out.write(struct.pack('<q', s))
    dtype_map = {
        np.dtype('complex128'): 0,
        np.dtype('float64'): 1,
        np.dtype('int32'): 2,
        np.dtype('uint8'): 3,
    }
    out.write(struct.pack('<i', dtype_map[arr.dtype]))
    out.write(arr.tobytes(order='F'))  # Fortran order for Julia column-major


def read_array(inp):
    """Read self-describing binary array from stream."""
    raw = inp.read(4)
    if len(raw) < 4:
        raise EOFError("Unexpected end of input")
    ndim = struct.unpack('<i', raw)[0]
    shape = tuple(struct.unpack('<q', inp.read(8))[0] for _ in range(ndim))
    dtype_code = struct.unpack('<i', inp.read(4))[0]
    dtype = {0: np.complex128, 1: np.float64, 2: np.int32, 3: np.uint8}[dtype_code]
    nbytes = int(np.prod(shape)) * dtype.itemsize
    raw_data = inp.read(nbytes)
    if len(raw_data) < nbytes:
        raise EOFError(f"Expected {nbytes} bytes, got {len(raw_data)}")
    # Reshape with Fortran order since Julia writes column-major
    return np.ascontiguousarray(np.frombuffer(raw_data, dtype=dtype).reshape(shape, order='F'))


def cmd_read(ms_path, column):
    """Read MS and write 8 binary arrays to stdout."""
    import casacore.tables as ct

    out = sys.stdout.buffer

    # Main table
    ms = ct.table(ms_path, readonly=True)
    data = np.ascontiguousarray(ms.getcol(column), dtype=np.complex128)
    flags = np.ascontiguousarray(ms.getcol("FLAG"), dtype=np.uint8)
    ant1 = np.ascontiguousarray(ms.getcol("ANTENNA1"), dtype=np.int32)
    ant2 = np.ascontiguousarray(ms.getcol("ANTENNA2"), dtype=np.int32)
    uvw = np.ascontiguousarray(ms.getcol("UVW"), dtype=np.float64)
    ms.close()

    # Spectral window
    spw = ct.table(ms_path + "/SPECTRAL_WINDOW", readonly=True)
    chan_freq = np.ascontiguousarray(spw.getcol("CHAN_FREQ"), dtype=np.float64)
    spw.close()

    # Antenna positions
    ant_tab = ct.table(ms_path + "/ANTENNA", readonly=True)
    positions = np.ascontiguousarray(ant_tab.getcol("POSITION"), dtype=np.float64)
    ant_tab.close()

    # Phase center
    field_tab = ct.table(ms_path + "/FIELD", readonly=True)
    phase_dir = np.ascontiguousarray(field_tab.getcol("PHASE_DIR"), dtype=np.float64)
    field_tab.close()

    # Write 8 arrays in fixed order
    write_array(out, data)
    write_array(out, flags)
    write_array(out, ant1)
    write_array(out, ant2)
    write_array(out, uvw)
    write_array(out, chan_freq)
    write_array(out, positions)
    write_array(out, phase_dir)
    out.flush()


def cmd_write(ms_path, column):
    """Read 2 binary arrays from stdin and write to MS."""
    import casacore.tables as ct

    inp = sys.stdin.buffer
    data = read_array(inp)
    flags = read_array(inp).astype(np.bool_)

    ms = ct.table(ms_path, readonly=False)
    ms.putcol(column, data)
    ms.putcol("FLAG", flags)
    ms.flush()
    ms.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    ms_path = sys.argv[2]
    column = sys.argv[3]

    if cmd == "read":
        cmd_read(ms_path, column)
    elif cmd == "write":
        cmd_write(ms_path, column)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
