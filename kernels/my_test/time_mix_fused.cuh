#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

/*
 * Fused Time-Mixing Kernel for RWKV
 * 
 * Computes:
 *   xx = prev - x
 *   xr = x + xx * x_r
 *   xw = x + xx * x_w
 *   xk = x + xx * x_k
 *   xv = x + xx * x_v
 *   xa = x + xx * x_a
 *   xg = x + xx * x_g
 *   prev = x  (carried to next timestep)
 *
 * Optimizations:
 *   - Weights loaded once to shared memory, reused across all T
 *   - prev carried in registers across iterations (no reload)
 *   - 6 outputs parallelized across warps (critical for T=1 decode)
 *   - Double-buffered x input for T>1 prefill
 */

template<int T_BLOCK>
struct time_mix_layout {
    using base_tile = st_bf<32, 32>;  // 1024 elements as 32x32 for TMA
    
    // Global memory descriptors
    struct globals {
        gl<bf16, 1, 1, -1, 32, base_tile> x;         // [B, T, 32, 32]
        gl<bf16, 1, 1, 1, 32, base_tile> x_prev;     // [B, 32, 32]
        // Weights: [32, 32] each, broadcasted across B and T
        gl<bf16, 1, 1, 1, 32, base_tile> x_r, x_w, x_k, x_v, x_a, x_g;
        // Outputs: [B, T, 32, 32] each
        gl<bf16, 1, 1, -1, 32, base_tile> out_r, out_w, out_k, out_v, out_a, out_g;
    };
    
    // Streamed input per iteration
    struct input_block {
        base_tile x;
    };
    
    // Persistent shared memory (not double-buffered)
    struct scratch_block {
        base_tile weights[6];   // x_r, x_w, x_k, x_v, x_a, x_g
        base_tile prev_smem;    // For init load and final store
    };
    
    // Output staging area
    struct finish_block {
        base_tile outputs[6];   // xr, xw, xk, xv, xa, xg
    };
    
    struct common_state {
        int batch_idx;
    };
    
    // Register-resident state carried across iterations
    struct consumer_state {
        rt_bf<32, 32> prev;     // 16 registers per warp
    };
};

template<int T_BLOCK = 1>
struct time_mix_template {
    using layout = time_mix_layout<T_BLOCK>;
    using base_tile = typename layout::base_tile;
    
    // 6 consumer warps: one per output for maximum T=1 parallelism
    // Warp i computes output i directly
    static constexpr int NUM_CONSUMER_WARPS = 6;
    static constexpr int INPUT_PIPE_STAGES = (T_BLOCK > 1) ? 2 : 1;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    
    __host__ static dim3 grid(int B, int T) {
        // One block per batch element; T_BLOCK timesteps per block
        return dim3((B * T + T_BLOCK - 1) / T_BLOCK);
    }
    
    __device__ static void common_setup(common_setup_args<layout> args) {
        int block_id = blockIdx.x;
        args.common.batch_idx = block_id;  // Simplified: 1 block per batch
        args.num_iters = T_BLOCK;
        
        // Producer warpgroup loads weights (once) and initial prev
        if (warpgroup::groupid() == 0) {
            int wid = warpgroup::warpid();
            
            // Distribute weight loads across producer warps
            // 4 warps, 6 weights + 1 prev = 7 loads
            if (wid == 0) {
                tma::load_async(args.scratch.weights[0], args.globals.x_r, {0, 0, 0});
                tma::load_async(args.scratch.weights[1], args.globals.x_w, {0, 0, 0});
            } else if (wid == 1) {
                tma::load_async(args.scratch.weights[2], args.globals.x_k, {0, 0, 0});
                tma::load_async(args.scratch.weights[3], args.globals.x_v, {0, 0, 0});
            } else if (wid == 2) {
                tma::load_async(args.scratch.weights[4], args.globals.x_a, {0, 0, 0});
                tma::load_async(args.scratch.weights[5], args.globals.x_g, {0, 0, 0});
            } else if (wid == 3) {
                tma::load_async(args.scratch.prev_smem, args.globals.x_prev, 
                               {args.common.batch_idx, 0, 0});
            }
            tma::load_async_wait();
        }
        __syncthreads();  // Ensure weights visible to consumers
    }
    
    __device__ static void consumer_setup(consumer_setup_args<layout> args) {
        // All consumer warps load prev to their own registers
        load(args.state.prev, args.scratch.prev_smem);
    }
    
    __device__ static void producer_setup(producer_setup_args<layout> args) {
        warpgroup::decrease_registers<32>();  // Producers need minimal registers
    }
    
    __device__ static void producer_load(producer_load_args<layout> args) {
        if (warpgroup::warpid() == 0) {
            int t = args.iter;
            tma::load_async(args.input.x, args.globals.x,
                           {args.common.batch_idx, t, 0, 0}, args.barrier);
        }
        if constexpr (PRODUCER_BARRIER_ARRIVALS > 0) {
            tma::arrive(args.barrier, PRODUCER_BARRIER_ARRIVALS);
        }
    }
    
    __device__ static void consumer_compute(consumer_compute_args<layout> args) {
        // Each of 6 warps handles one output independently
        // This is the key optimization for T=1: all 6 outputs computed in parallel
        
        int warp = warpgroup::warpid();
        
        // All warps need x and xx
        rt_bf<32, 32> x_reg, xx;
        load(x_reg, args.input.x);
        sub(xx, args.state.prev, x_reg);  // xx = prev - x
        
        // Each warp computes its assigned output
        // warp 0 -> xr, warp 1 -> xw, ..., warp 5 -> xg
        if (warp < 6) {
            rt_bf<32, 32> weight, result;
            load(weight, args.scratch.weights[warp]);
            mul(result, xx, weight);           // xx * weight
            add(result, x_reg, result);        // x + xx * weight
            store(args.finish.outputs[warp], result);
        }
        
        // Update prev for next iteration (all warps, for consistency)
        copy(args.state.prev, x_reg);
    }
    
    __device__ static void consumer_finish(consumer_finish_args<layout> args) {
        int t = args.iter;
        int b = args.common.batch_idx;
        int warp = warpgroup::warpid();
        
        // Each warp stores its output via TMA
        // This parallelizes the 6 stores for T=1 efficiency
        if (warp == 0) {
            tma::store_async(args.globals.out_r, args.finish.outputs[0], {b, t, 0, 0});
        } else if (warp == 1) {
            tma::store_async(args.globals.out_w, args.finish.outputs[1], {b, t, 0, 0});
        } else if (warp == 2) {
            tma::store_async(args.globals.out_k, args.finish.outputs[2], {b, t, 0, 0});
        } else if (warp == 3) {
            tma::store_async(args.globals.out_v, args.finish.outputs[3], {b, t, 0, 0});
        } else if (warp == 4) {
            tma::store_async(args.globals.out_a, args.finish.outputs[4], {b, t, 0, 0});
        } else if (warp == 5) {
            tma::store_async(args.globals.out_g, args.finish.outputs[5], {b, t, 0, 0});
        }
        
        // On final iteration, write back updated x_prev
        if (args.iter == args.num_iters - 1 && warp == 0) {
            // prev is in registers, need to go through shared memory for TMA
            store(args.scratch.prev_smem, args.state.prev);
            __syncwarp();  // Ensure store completes
            tma::store_async(args.globals.x_prev, args.scratch.prev_smem, {b, 0, 0});
        }
        
        tma::store_async_wait();
    }
};

// Explicit instantiations for common cases
using time_mix_decode  = time_mix_template<1>;   // T=1 decode
using time_mix_prefill = time_mix_template<16>;  // T=16 prefill chunk

// Globals struct for pybind11 binding
template<int T_BLOCK>
struct time_mix_globals {
    using layout = time_mix_layout<T_BLOCK>;
    using base_tile = typename layout::base_tile;
    
    gl<bf16, 1, 1, -1, 32, base_tile> x;
    gl<bf16, 1, 1, 1, 32, base_tile> x_prev;
    gl<bf16, 1, 1, 1, 32, base_tile> x_r, x_w, x_k, x_v, x_a, x_g;
    gl<bf16, 1, 1, -1, 32, base_tile> out_r, out_w, out_k, out_v, out_a, out_g;
    
    dim3 grid() {
        return time_mix_template<T_BLOCK>::grid(x.batch, x.depth);
    }
    dim3 block() {
        return dim3(prototype::detail::NUM_THREADS);
    }
};

using decode_globals  = time_mix_globals<1>;
using prefill_globals = time_mix_globals<16>;

void run_time_mix_decode(decode_globals g) {
    using kernel = time_mix_template<1>;
    using layout = typename kernel::layout;
    
    // Convert to layout::globals format
    typename layout::globals lg{
        .x = g.x,
        .x_prev = g.x_prev,
        .x_r = g.x_r, .x_w = g.x_w, .x_k = g.x_k,
        .x_v = g.x_v, .x_a = g.x_a, .x_g = g.x_g,
        .out_r = g.out_r, .out_w = g.out_w, .out_k = g.out_k,
        .out_v = g.out_v, .out_a = g.out_a, .out_g = g.out_g
    };
    
    prototype::lcf::kernel<kernel><<<g.grid(), g.block()>>>(lg);
}

void run_time_mix_prefill(prefill_globals g) {
    using kernel = time_mix_template<16>;
    using layout = typename kernel::layout;
    
    typename layout::globals lg{
        .x = g.x,
        .x_prev = g.x_prev,
        .x_r = g.x_r, .x_w = g.x_w, .x_k = g.x_k,
        .x_v = g.x_v, .x_a = g.x_a, .x_g = g.x_g,
        .out_r = g.out_r, .out_w = g.out_w, .out_k = g.out_k,
        .out_v = g.out_v, .out_a = g.out_a, .out_g = g.out_g
    };
    
    prototype::lcf::kernel<kernel><<<g.grid(), g.block()>>>(lg);
}

PYBIND11_MODULE(time_mix_tk, m) {
    m.doc() = "ThunderKittens fused time-mixing kernel for RWKV";
    
    // Decode kernel (T=1)
    py::bind_function<run_time_mix_decode>(m, "time_mix_decode",
        &decode_globals::x, &decode_globals::x_prev,
        &decode_globals::x_r, &decode_globals::x_w, &decode_globals::x_k,
        &decode_globals::x_v, &decode_globals::x_a, &decode_globals::x_g,
        &decode_globals::out_r, &decode_globals::out_w, &decode_globals::out_k,
        &decode_globals::out_v, &decode_globals::out_a, &decode_globals::out_g);
    
    // Prefill kernel (T=16 chunks)
    py::bind_function<run_time_mix_prefill>(m, "time_mix_prefill",
        &prefill_globals::x, &prefill_globals::x_prev,
        &prefill_globals::x_r, &prefill_globals::x_w, &prefill_globals::x_k,
        &prefill_globals::x_v, &prefill_globals::x_a, &prefill_globals::x_g,
        &prefill_globals::out_r, &prefill_globals::out_w, &prefill_globals::out_k,
        &prefill_globals::out_v, &prefill_globals::out_a, &prefill_globals::out_g);
}
