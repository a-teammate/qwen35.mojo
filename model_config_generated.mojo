from std.math import sqrt

comptime DIM = 1024
comptime N_LAYERS = 24
comptime FFN_DIM = 3584
comptime VOCAB_SIZE = 248320
comptime SEQ_LEN = 262144
comptime ROPE_THETA = 10000000.0
comptime RMS_NORM_EPS = 1e-6

comptime FULL_HEAD_DIM = 256
comptime FULL_N_Q_HEADS = 8
comptime FULL_N_KV_HEADS = 2
comptime FULL_Q_DIM = FULL_N_Q_HEADS * FULL_HEAD_DIM
comptime FULL_KV_DIM = FULL_N_KV_HEADS * FULL_HEAD_DIM
comptime FULL_Q_TOTAL_DIM = FULL_Q_DIM * 2
comptime ROTARY_DIM = FULL_HEAD_DIM / 4
comptime FULL_GQA_RATIO = FULL_N_Q_HEADS / FULL_N_KV_HEADS
comptime FULL_ATTENTION_INTERVAL = 4
comptime N_FULL_ATTN_LAYERS = N_LAYERS / FULL_ATTENTION_INTERVAL

comptime LINEAR_N_HEADS = 16
comptime LINEAR_HEAD_DIM = 128
comptime LINEAR_Q_DIM = LINEAR_N_HEADS * LINEAR_HEAD_DIM
comptime LINEAR_KV_DIM = LINEAR_N_HEADS * LINEAR_HEAD_DIM
comptime LINEAR_FUSED_DIM = LINEAR_Q_DIM + LINEAR_KV_DIM * 2
comptime LINEAR_CONV_KERNEL = 4
comptime LINEAR_STATE_HEADS = 16
comptime LINEAR_STATE_DIM = LINEAR_HEAD_DIM * LINEAR_HEAD_DIM
comptime N_DELTANET_LAYERS = 18

comptime KV_CACHE_CAP = 4096
comptime KV_CACHE_SIZE = KV_CACHE_CAP * FULL_N_KV_HEADS * FULL_HEAD_DIM * 2 * 6
comptime DELTANET_STATE_SIZE = N_DELTANET_LAYERS * LINEAR_STATE_HEADS * LINEAR_STATE_DIM
comptime CONV_BUF_SIZE = LINEAR_CONV_KERNEL * LINEAR_FUSED_DIM * N_DELTANET_LAYERS

fn is_deltanet(layer: Int) -> Bool:
    return layer % 4 != 3

fn is_full_attention(layer: Int) -> Bool:
    return layer % 4 == 3

fn deltanet_layer_index(layer: Int) -> Int:
    return layer - layer // 4

fn full_attn_layer_index(layer: Int) -> Int:
    return layer // 4

fn deltanet_state_offset(layer: Int) -> Int:
    return deltanet_layer_index(layer) * LINEAR_STATE_HEADS * LINEAR_STATE_DIM

fn conv_buf_offset(layer: Int) -> Int:
    return deltanet_layer_index(layer) * LINEAR_CONV_KERNEL * LINEAR_FUSED_DIM

fn full_attn_kv_offset(layer: Int) -> Int:
    return full_attn_layer_index(layer) * KV_CACHE_CAP * FULL_N_KV_HEADS * FULL_HEAD_DIM * 2
