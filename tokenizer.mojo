from collections.list import List
from collections.dict import Dict
from gguf_loader import (
    GGUFData,
    gguf_get_metadata_uint32,
    gguf_get_metadata_string_array,
)
from std.python import Python

fn _build_byte_to_unicode() -> List[String]:
    var byte_to_char = List[String]()
    for _ in range(256):
        byte_to_char.append("")
    var n = 0
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
            byte_to_char[b] = String(chr(b))
        else:
            byte_to_char[b] = String(chr(256 + n))
            n += 1
    return byte_to_char^

fn _build_char_to_byte() -> Dict[Int, Int]:
    var mapping = Dict[Int, Int]()
    var n = 0
    for b in range(256):
        if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
            mapping[b] = b
        else:
            mapping[256 + n] = b
            n += 1
    return mapping^

struct Tokenizer:
    var vocab: List[String]
    var vocab_to_id: Dict[String, Int]
    var merge_rank: Dict[String, Int]
    var byte_to_token: List[Int]
    var char_to_byte: Dict[Int, Int]
    var bos_id: Int
    var eos_id: Int

    def __init__(out self, gguf: GGUFData) raises:
        self.vocab_to_id = Dict[String, Int]()
        self.merge_rank = Dict[String, Int]()
        self.byte_to_token = List[Int]()
        self.char_to_byte = _build_char_to_byte()
        self.bos_id = gguf_get_metadata_uint32(gguf, "tokenizer.ggml.bos_token_id")
        self.eos_id = gguf_get_metadata_uint32(gguf, "tokenizer.ggml.eos_token_id")

        self.vocab = gguf_get_metadata_string_array(gguf, "tokenizer.ggml.tokens")
        for i in range(len(self.vocab)):
            self.vocab_to_id[self.vocab[i]] = i

        var merges = gguf_get_metadata_string_array(gguf, "tokenizer.ggml.merges")
        for rank in range(len(merges)):
            self.merge_rank[merges[rank]] = rank

        var byte_to_char = _build_byte_to_unicode()
        for b in range(256):
            var tid = self.find(byte_to_char[b])
            self.byte_to_token.append(tid)

    def find(self, token: String) -> Int:
        var idx = self.vocab_to_id.find(token)
        if idx: return idx.value()
        return -1

    def _bpe_encode_chunk(self, text: String) -> List[Int]:
        var tokens = List[Int]()
        var ptr = text.unsafe_ptr()
        for i in range(len(text)):
            var b = Int(ptr[i])
            var tid = self.byte_to_token[b]
            if tid == -1:
                tid = 0
            tokens.append(tid)

        while True:
            var best_rank = len(self.merge_rank) + 1
            var best_idx = -1
            for i in range(len(tokens) - 1):
                var pair = self.vocab[tokens[i]] + " " + self.vocab[tokens[i + 1]]
                var rank = self.merge_rank.find(pair)
                if rank and rank.value() < best_rank:
                    best_rank = rank.value()
                    best_idx = i
            if best_idx == -1:
                break
            var merged_token = self.vocab[tokens[best_idx]] + self.vocab[tokens[best_idx + 1]]
            var merged_id = self.find(merged_token)
            if merged_id == -1:
                break
            var new_tokens = List[Int]()
            for i in range(best_idx):
                new_tokens.append(tokens[i])
            new_tokens.append(merged_id)
            for i in range(best_idx + 2, len(tokens)):
                new_tokens.append(tokens[i])
            tokens = new_tokens^

        return tokens^

    def encode(self, text: String) raises -> List[Int]:
        var re_mod = Python.import_module("regex")
        var pat = String(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
            "\\p{N}|"
            " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|"
            "\\s+(?!\\S)|"
            "\\s+"
        )
        var chunks = re_mod.findall(pat, text)
        var all_tokens = List[Int]()
        var n_chunks = Int(chunks.__len__())
        for i in range(n_chunks):
            var chunk_py = chunks[i].__str__()
            var chunk = String(Python.str(chunk_py))
            var tokens = self._bpe_encode_chunk(chunk)
            for j in range(len(tokens)):
                all_tokens.append(tokens[j])
        return all_tokens^

    def _is_hex_byte_token(self, token: String) -> Bool:
        if len(token) < 5: return False
        var ptr = token.unsafe_ptr()
        return ptr[0] == 60 and ptr[1] == 48 and ptr[2] == 120

    def decode(self, token_id: Int) -> String:
        if token_id < 0 or token_id >= len(self.vocab):
            return ""
        var token_str = self.vocab[token_id]
        if self._is_hex_byte_token(token_str):
            return self._decode_hex_byte_token(token_str)
        var bytes = List[Int]()
        var ptr = token_str.unsafe_ptr()
        var pos = 0
        var slen = len(token_str)
        while pos < slen:
            var b0 = Int(ptr[pos])
            var cp = b0
            var cplen = 1
            if b0 >= 224 and b0 < 240 and pos + 2 < slen:
                cp = ((b0 & 15) << 12) | ((Int(ptr[pos + 1]) & 63) << 6) | (Int(ptr[pos + 2]) & 63)
                cplen = 3
            elif b0 >= 192 and b0 < 224 and pos + 1 < slen:
                cp = ((b0 & 31) << 6) | (Int(ptr[pos + 1]) & 63)
                cplen = 2
            elif b0 >= 240 and pos + 3 < slen:
                cp = ((b0 & 7) << 18) | ((Int(ptr[pos + 1]) & 63) << 12) | ((Int(ptr[pos + 2]) & 63) << 6) | (Int(ptr[pos + 3]) & 63)
                cplen = 4
            var byte_val = self.char_to_byte.find(cp)
            if byte_val:
                bytes.append(byte_val.value())
            pos += cplen
        return self._bytes_to_string(bytes)

    def decode_all(self, token_ids: List[Int]) -> String:
        var result = ""
        for i in range(len(token_ids)):
            result += self.decode(token_ids[i])
        return result

    def _decode_hex_byte_token(self, token: String) -> String:
        if len(token) < 5: return ""
        var ptr = token.unsafe_ptr()
        var val = 0
        for i in range(3, len(token) - 1):
            var c = Int(ptr[i])
            val = val * 16
            if c >= 48 and c <= 57: val += c - 48
            elif c >= 97 and c <= 102: val += c - 87
            elif c >= 65 and c <= 70: val += c - 55
        return String(chr(val))

    def _bytes_to_string(self, bytes: List[Int]) -> String:
        var result = ""
        var i = 0
        while i < len(bytes):
            var b0 = bytes[i]
            if b0 < 128:
                result += String(chr(b0))
                i += 1
            elif b0 < 224:
                if i + 1 < len(bytes):
                    var cp = ((b0 & 31) << 6) | (bytes[i + 1] & 63)
                    result += String(chr(cp))
                    i += 2
                else:
                    result += String(chr(b0))
                    i += 1
            elif b0 < 240:
                if i + 2 < len(bytes):
                    var cp = ((b0 & 15) << 12) | ((bytes[i + 1] & 63) << 6) | (bytes[i + 2] & 63)
                    result += String(chr(cp))
                    i += 3
                else:
                    result += String(chr(b0))
                    i += 1
            else:
                if i + 3 < len(bytes):
                    var cp = ((b0 & 7) << 18) | ((bytes[i + 1] & 63) << 12) | ((bytes[i + 2] & 63) << 6) | (bytes[i + 3] & 63)
                    result += String(chr(cp))
                    i += 4
                else:
                    result += String(chr(b0))
                    i += 1
        return result
