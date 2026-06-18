from std.memory import alloc, memset_zero, memcpy
from std.memory.unsafe import bitcast as unsafe_bitcast
from collections.list import List
from std.io.file import open as file_open
from std.os.fstat import stat as file_stat

@always_inline
fn _read_u32(data: UnsafePointer[UInt8, ImmutExternalOrigin], off: Int) -> Int:
    return Int(data[off]) | (Int(data[off + 1]) << 8) | (Int(data[off + 2]) << 16) | (Int(data[off + 3]) << 24)

@always_inline
fn _read_u64(data: UnsafePointer[UInt8, ImmutExternalOrigin], off: Int) -> Int:
    var lo = Int(data[off]) | (Int(data[off + 1]) << 8) | (Int(data[off + 2]) << 16) | (Int(data[off + 3]) << 24)
    var hi = Int(data[off + 4]) | (Int(data[off + 5]) << 8) | (Int(data[off + 6]) << 16) | (Int(data[off + 7]) << 24)
    return lo | (hi << 32)

@always_inline
fn _read_bytes_as_string(data: UnsafePointer[UInt8, ImmutExternalOrigin], off: Int, length: Int) -> String:
    var result = ""
    var i = 0
    while i < length:
        var b = Int(data[off + i])
        if b < 128:
            result += String(chr(b))
            i += 1
        elif b < 224:
            var cp = ((b & 0x1F) << 6) | (Int(data[off + i + 1]) & 0x3F)
            result += String(chr(cp))
            i += 2
        elif b < 240:
            var cp = ((b & 0x0F) << 12) | ((Int(data[off + i + 1]) & 0x3F) << 6) | (Int(data[off + i + 2]) & 0x3F)
            result += String(chr(cp))
            i += 3
        else:
            var cp = ((b & 0x07) << 18) | ((Int(data[off + i + 1]) & 0x3F) << 12) | ((Int(data[off + i + 2]) & 0x3F) << 6) | (Int(data[off + i + 3]) & 0x3F)
            result += String(chr(cp))
            i += 4
    return result

fn _skip_value(data: UnsafePointer[UInt8, ImmutExternalOrigin], off_in: Int, vtype: Int) -> Int:
    var off = off_in
    if vtype <= 1: return off + 1
    if vtype <= 3: return off + 2
    if vtype <= 6: return off + 4
    if vtype == 7: return off + 1
    if vtype == 8:
        var slen = _read_u64(data, off)
        return off + 8 + slen
    if vtype == 9:
        var elem_type = _read_u32(data, off)
        off += 4
        var count = _read_u64(data, off)
        off += 8
        if elem_type == 8:
            for _ in range(count):
                var slen = _read_u64(data, off)
                off += 8 + slen
        elif elem_type <= 1:
            off += count
        elif elem_type <= 3:
            off += count * 2
        elif elem_type <= 6:
            off += count * 4
        elif elem_type == 7:
            off += count
        elif elem_type >= 10:
            off += count * 8
        return off
    if vtype >= 10: return off + 8
    return off

struct GGUFData:
    var data: UnsafePointer[UInt8, ImmutExternalOrigin]
    var total_size: Int
    var version: Int
    var tensor_count: Int
    var metadata_kv_count: Int
    var data_start: Int
    var tensor_names: UnsafePointer[Int, MutExternalOrigin]
    var tensor_name_lens: UnsafePointer[Int, MutExternalOrigin]
    var tensor_types: UnsafePointer[Int, MutExternalOrigin]
    var tensor_dim0s: UnsafePointer[Int, MutExternalOrigin]
    var tensor_dim1s: UnsafePointer[Int, MutExternalOrigin]
    var tensor_offsets: UnsafePointer[Int, MutExternalOrigin]
    var md_key_offsets: UnsafePointer[Int, MutExternalOrigin]
    var md_key_lens: UnsafePointer[Int, MutExternalOrigin]
    var md_value_types: UnsafePointer[Int, MutExternalOrigin]
    var md_value_offsets: UnsafePointer[Int, MutExternalOrigin]
    var md_count: Int

    def __init__(out self,
        data: UnsafePointer[UInt8, ImmutExternalOrigin],
        total_size: Int,
        version: Int,
        tensor_count: Int,
        metadata_kv_count: Int,
        data_start: Int,
        tensor_names: UnsafePointer[Int, MutExternalOrigin],
        tensor_name_lens: UnsafePointer[Int, MutExternalOrigin],
        tensor_types: UnsafePointer[Int, MutExternalOrigin],
        tensor_dim0s: UnsafePointer[Int, MutExternalOrigin],
        tensor_dim1s: UnsafePointer[Int, MutExternalOrigin],
        tensor_offsets: UnsafePointer[Int, MutExternalOrigin],
        md_key_offsets: UnsafePointer[Int, MutExternalOrigin],
        md_key_lens: UnsafePointer[Int, MutExternalOrigin],
        md_value_types: UnsafePointer[Int, MutExternalOrigin],
        md_value_offsets: UnsafePointer[Int, MutExternalOrigin],
        md_count: Int,
    ):
        self.data = data
        self.total_size = total_size
        self.version = version
        self.tensor_count = tensor_count
        self.metadata_kv_count = metadata_kv_count
        self.data_start = data_start
        self.tensor_names = tensor_names
        self.tensor_name_lens = tensor_name_lens
        self.tensor_types = tensor_types
        self.tensor_dim0s = tensor_dim0s
        self.tensor_dim1s = tensor_dim1s
        self.tensor_offsets = tensor_offsets
        self.md_key_offsets = md_key_offsets
        self.md_key_lens = md_key_lens
        self.md_value_types = md_value_types
        self.md_value_offsets = md_value_offsets
        self.md_count = md_count

fn gguf_load(path: String) raises -> GGUFData:
    var st = file_stat(path)
    var total_size = Int(st.st_size)

    var f = file_open(path, "r")
    var content = f.read_bytes(total_size)
    f.close()

    var buf = alloc[UInt8](total_size)
    memcpy(dest=buf, src=content.unsafe_ptr(), count=total_size)
    var data = UnsafePointer[UInt8, ImmutExternalOrigin](buf.address)

    var version = _read_u32(data, 4)
    var tensor_count = _read_u64(data, 8)
    var metadata_kv_count = _read_u64(data, 16)

    print("GGUF version:", version, "tensors:", tensor_count, "metadata_kvs:", metadata_kv_count)

    var md_key_offsets = alloc[Int](metadata_kv_count)
    var md_key_lens = alloc[Int](metadata_kv_count)
    var md_value_types = alloc[Int](metadata_kv_count)
    var md_value_offsets = alloc[Int](metadata_kv_count)

    var off: Int = 24
    for i in range(metadata_kv_count):
        var key_len = _read_u64(data, off)
        off += 8
        md_key_offsets[i] = off
        md_key_lens[i] = key_len
        off += key_len

        var vtype = _read_u32(data, off)
        off += 4
        md_value_types[i] = vtype
        md_value_offsets[i] = off

        off = _skip_value(data, off, vtype)

    var tensor_names = alloc[Int](tensor_count)
    var tensor_name_lens = alloc[Int](tensor_count)
    var tensor_types = alloc[Int](tensor_count)
    var tensor_dim0s = alloc[Int](tensor_count)
    var tensor_dim1s = alloc[Int](tensor_count)
    var tensor_offsets = alloc[Int](tensor_count)

    for t in range(tensor_count):
        var name_len = _read_u64(data, off)
        off += 8
        tensor_names[t] = off
        tensor_name_lens[t] = name_len
        off += name_len
        var n_dims = _read_u32(data, off)
        off += 4
        tensor_dim0s[t] = 0
        tensor_dim1s[t] = 0
        if n_dims > 0:
            tensor_dim0s[t] = _read_u64(data, off)
            off += 8
        if n_dims > 1:
            tensor_dim1s[t] = _read_u64(data, off)
            off += 8
        for _d in range(max(0, n_dims - 2)):
            off += 8
        tensor_types[t] = _read_u32(data, off)
        off += 4
        tensor_offsets[t] = _read_u64(data, off)
        off += 8

    var data_start = (off + 31) & ~31

    for t in range(tensor_count):
        var name = _read_bytes_as_string(data, tensor_names[t], tensor_name_lens[t])
        print("  [", t, "] ", name, " type=", tensor_types[t],
              " dims=[", tensor_dim0s[t], ",", tensor_dim1s[t],
              "] offset=", tensor_offsets[t])

    return GGUFData(
        data=data,
        total_size=total_size,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        data_start=data_start,
        tensor_names=tensor_names,
        tensor_name_lens=tensor_name_lens,
        tensor_types=tensor_types,
        tensor_dim0s=tensor_dim0s,
        tensor_dim1s=tensor_dim1s,
        tensor_offsets=tensor_offsets,
        md_key_offsets=md_key_offsets,
        md_key_lens=md_key_lens,
        md_value_types=md_value_types,
        md_value_offsets=md_value_offsets,
        md_count=metadata_kv_count,
    )

fn gguf_find_tensor(gguf: GGUFData, name: String) -> Int:
    for t in range(gguf.tensor_count):
        if _read_bytes_as_string(gguf.data, gguf.tensor_names[t], gguf.tensor_name_lens[t]) == name:
            return t
    return -1

fn gguf_tensor_data_ptr(gguf: GGUFData, idx: Int) -> UnsafePointer[UInt8, ImmutExternalOrigin]:
    return gguf.data + gguf.data_start + gguf.tensor_offsets[idx]

fn gguf_find_metadata(gguf: GGUFData, key: String) -> Int:
    for i in range(gguf.md_count):
        var k = _read_bytes_as_string(gguf.data, gguf.md_key_offsets[i], gguf.md_key_lens[i])
        if k == key:
            return i
    return -1

fn gguf_get_metadata_string(gguf: GGUFData, key: String) -> String:
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return ""
    if gguf.md_value_types[idx] != 8: return ""
    var off = gguf.md_value_offsets[idx]
    var slen = _read_u64(gguf.data, off)
    return _read_bytes_as_string(gguf.data, off + 8, slen)

fn gguf_get_metadata_uint32(gguf: GGUFData, key: String) -> Int:
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return -1
    if gguf.md_value_types[idx] != 4: return -1
    return _read_u32(gguf.data, gguf.md_value_offsets[idx])

fn gguf_get_metadata_uint64(gguf: GGUFData, key: String) -> Int:
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return -1
    if gguf.md_value_types[idx] != 10: return -1
    return _read_u64(gguf.data, gguf.md_value_offsets[idx])

fn gguf_get_metadata_float64(gguf: GGUFData, key: String) -> Float64:
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return 0.0
    if gguf.md_value_types[idx] != 12: return 0.0
    return Float64((gguf.data + gguf.md_value_offsets[idx]).bitcast[Float64]()[0])

fn gguf_get_metadata_array_count(gguf: GGUFData, key: String) -> Int:
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return 0
    if gguf.md_value_types[idx] != 9: return 0
    var off = gguf.md_value_offsets[idx]
    off += 4
    return _read_u64(gguf.data, off)

fn fp16_to_f32(bits: UInt16) -> Float32:
    var v = SIMD[DType.uint16, 1](bits)
    var f16 = unsafe_bitcast[DType.float16](v)
    return Float32(f16[0])

fn dequant_q8_0(
    src: UnsafePointer[UInt8, ImmutExternalOrigin],
    n_weights: Int,
) -> UnsafePointer[Float32, MutExternalOrigin]:
    var dst = alloc[Float32](n_weights)
    var n_blocks = n_weights // 32
    var remainder = n_weights % 32
    var b = 0
    while b < n_blocks:
        var block_ptr = src + b * 34
        var scale_bits = UInt16(Int(block_ptr[0]) | (Int(block_ptr[1]) << 8))
        var scale = fp16_to_f32(scale_bits)
        var qs_ptr = block_ptr + 2
        var base = b * 32
        for i in range(32):
            var raw = Int(qs_ptr[i])
            var signed_val = (raw ^ 0x80) - 0x80
            dst[base + i] = scale * Float32(signed_val)
        b += 1
    if remainder > 0:
        var block_ptr = src + n_blocks * 34
        var scale_bits = UInt16(Int(block_ptr[0]) | (Int(block_ptr[1]) << 8))
        var scale = fp16_to_f32(scale_bits)
        var qs_ptr = block_ptr + 2
        var base = n_blocks * 32
        for i in range(remainder):
            var raw = Int(qs_ptr[i])
            var signed_val = (raw ^ 0x80) - 0x80
            dst[base + i] = scale * Float32(signed_val)
    return dst

fn repack_q8_0_tiled(
    src: UnsafePointer[UInt8, ImmutExternalOrigin],
    dst: UnsafePointer[UInt8, MutExternalOrigin],
    n_rows: Int,
    n_blocks: Int,
    tile_size: Int,
):
    var tile_start = 0
    var tile_idx = 0
    while tile_start < n_rows:
        var row_in_tile = 0
        while row_in_tile < tile_size:
            var src_row = tile_start + row_in_tile
            var b = 0
            while b < n_blocks:
                var src_off = (src_row * n_blocks + b) * 34
                var dst_off = (tile_idx * n_blocks + b) * tile_size * 34 + row_in_tile * 34
                var i = 0
                while i < 34:
                    dst[dst_off + i] = src[src_off + i]
                    i += 1
                b += 1
            row_in_tile += 1
        tile_start += tile_size
        tile_idx += 1


fn gguf_resolve_f32(
    gguf: GGUFData,
    name: String,
) -> UnsafePointer[Float32, ImmutExternalOrigin]:
    var idx = gguf_find_tensor(gguf, name)
    var ttype = gguf.tensor_types[idx]
    var raw_ptr = gguf_tensor_data_ptr(gguf, idx)
    if ttype == 0:
        return raw_ptr.bitcast[Float32]()
    if ttype == 8:
        var d0 = gguf.tensor_dim0s[idx]
        var d1 = gguf.tensor_dim1s[idx]
        var n_weights = d0
        if d1 > 0:
            n_weights = d0 * d1
        var dst = dequant_q8_0(raw_ptr, n_weights)
        return UnsafePointer[Float32, ImmutExternalOrigin](dst.address)
    print("  ERROR: unsupported tensor type ", ttype, " for ", name)
    return raw_ptr.bitcast[Float32]()

fn gguf_get_metadata_string_array(gguf: GGUFData, key: String) -> List[String]:
    var result = List[String]()
    var idx = gguf_find_metadata(gguf, key)
    if idx == -1: return result^
    if gguf.md_value_types[idx] != 9: return result^
    var off = gguf.md_value_offsets[idx]
    var elem_type = _read_u32(gguf.data, off)
    off += 4
    if elem_type != 8: return result^
    var count = _read_u64(gguf.data, off)
    off += 8
    for _ in range(count):
        var slen = _read_u64(gguf.data, off)
        off += 8
        result.append(_read_bytes_as_string(gguf.data, off, slen))
        off += slen
    return result^
