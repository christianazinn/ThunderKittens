// timemix_binding.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// ============================================================================
// LAYOUT DEFINITION
// ============================================================================

struct timemix_layout {
    using base_tile = st_bf<32, 32>;
    using global_layout = gl<bf16, -1, -1, 32, 32, base_tile>;
    
    struct globals {
        global_layout x;
        global_layout x_prev;
        global_layout x_r;
        global_layout x_w;
        global_layout x_k;
        global_layout x_v;
        global_layout x_a;
        global_layout x_g;
        global_layout xr_out;
        global_layout xw_out;
        global_layout xk_out;
        global_layout xv_out;
        global_layout xa_out;
        global_layout xg_out;
        
        dim3 grid()  { 
            int batch = x.batch();
            int T = x.cols();
            return dim3(batch * T);
        }
        dim3 block() { return dim3(64); }
    };
    
    struct input_block {
        base_tile x;
        base_tile x_prev;
        base_tile weights[6];
    };
    
    struct scratch_block {
        base_tile xx;  // Store xx here to reuse
        base_tile outputs_temp[6];
    };
    
    struct finish_block {
        base_tile outputs[6];
    };
    
    struct common_state {
        int batch_idx;
        int timestep;
    };
    
    struct consumer_state {
        // Use smaller 16x16 tiles and process in 4 chunks
        rt_bf<16, 16> x_chunk;
        rt_bf<16, 16> xx_chunk;
        rt_bf<16, 16> weight_chunk;
        rt_bf<16, 16> result_chunk;
    };
};

// ============================================================================
// KERNEL TEMPLATE
// ============================================================================

struct timemix_kernel_template {
    using layout = timemix_layout;
    
    static constexpr int NUM_CONSUMER_WARPS = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int T = args.globals.x.cols();
        int total_task = blockIdx.x;
        
        args.common.batch_idx = total_task / T;
        args.common.timestep = total_task % T;
        args.num_iters = 1;
    }
    
    // ========================================================================
    // PRODUCER
    // ========================================================================
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                int batch = args.common.batch_idx;
                int t = args.common.timestep;
                
                if (args.iter == 0) {
                    tma::expect(args.inputs_arrived, args.input, 8);
                    
                    tma::load_async(args.input.x, args.globals.x,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.x_prev, args.globals.x_prev,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                    
                    tma::load_async(args.input.weights[0], args.globals.x_r,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.weights[1], args.globals.x_w,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.weights[2], args.globals.x_k,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.weights[3], args.globals.x_v,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.weights[4], args.globals.x_a,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.weights[5], args.globals.x_g,
                                   {0, 0, 0, 0}, args.inputs_arrived);
                } else {
                    tma::expect(args.inputs_arrived, args.input, 2);
                    
                    tma::load_async(args.input.x, args.globals.x,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.x_prev, args.globals.x_prev,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                }
            }
        }
    };
    
    // ========================================================================
    // CONSUMER - Work in chunks to reduce register pressure
    // ========================================================================
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            // No register reallocation needed
        }
        
        __device__ static void compute(consumer_compute_args<layout> args) {
            // Process the 32x32 tiles in 4 chunks of 16x16
            // This dramatically reduces register usage
            
            // First, compute xx = x_prev - x and store in shared memory
            for (int chunk_row = 0; chunk_row < 2; chunk_row++) {
                for (int chunk_col = 0; chunk_col < 2; chunk_col++) {
                    // Load 16x16 chunks from 32x32 tiles
                    auto x_subtile = subtile<16, 16>(args.input.x, {chunk_row, chunk_col});
                    auto xprev_subtile = subtile<16, 16>(args.input.x_prev, {chunk_row, chunk_col});
                    auto xx_subtile = subtile<16, 16>(args.scratch.xx, {chunk_row, chunk_col});
                    
                    warp::load(args.state.x_chunk, x_subtile);
                    warp::load(args.state.xx_chunk, xprev_subtile);
                    
                    // xx = x_prev - x
                    warp::sub(args.state.xx_chunk, args.state.xx_chunk, args.state.x_chunk);
                    
                    // Store xx back to shared memory for reuse
                    warp::store(xx_subtile, args.state.xx_chunk);
                }
            }
            
            // Now compute all 6 outputs
            for (int output_idx = 0; output_idx < 6; output_idx++) {
                for (int chunk_row = 0; chunk_row < 2; chunk_row++) {
                    for (int chunk_col = 0; chunk_col < 2; chunk_col++) {
                        // Get subtiles
                        auto x_subtile = subtile<16, 16>(args.input.x, {chunk_row, chunk_col});
                        auto xx_subtile = subtile<16, 16>(args.scratch.xx, {chunk_row, chunk_col});
                        auto weight_subtile = subtile<16, 16>(args.input.weights[output_idx], {chunk_row, chunk_col});
                        auto result_subtile = subtile<16, 16>(args.scratch.outputs_temp[output_idx], {chunk_row, chunk_col});
                        
                        // Load chunks
                        warp::load(args.state.x_chunk, x_subtile);
                        warp::load(args.state.xx_chunk, xx_subtile);
                        warp::load(args.state.weight_chunk, weight_subtile);
                        
                        // result = xx * weight
                        warp::mul(args.state.result_chunk, 
                                 args.state.xx_chunk, 
                                 args.state.weight_chunk);
                        
                        // result = x + (xx * weight)
                        warp::add(args.state.result_chunk,
                                 args.state.x_chunk,
                                 args.state.result_chunk);
                        
                        // Store result chunk
                        warp::store(result_subtile, args.state.result_chunk);
                    }
                }
            }
            
            if (laneid() == 0) arrive(args.inputs_finished);
        }
        
        __device__ static void finish(consumer_finish_args<layout> args) {
            // Copy from scratch to finish
            for (int i = 0; i < 6; i++) {
                args.finish.outputs[i] = args.scratch.outputs_temp[i];
            }
            
            // Store all outputs
            int batch = args.common.batch_idx;
            int t = args.common.timestep;
            
            tma::store_async(args.globals.xr_out, args.finish.outputs[0],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            tma::store_async(args.globals.xw_out, args.finish.outputs[1],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            tma::store_async(args.globals.xk_out, args.finish.outputs[2],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            tma::store_async(args.globals.xv_out, args.finish.outputs[3],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            tma::store_async(args.globals.xa_out, args.finish.outputs[4],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            tma::store_async(args.globals.xg_out, args.finish.outputs[5],
                            {batch, t, 0, 0});
            tma::store_async_read_wait();
            
            if (laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// ============================================================================
// WRAPPER FUNCTION
// ============================================================================

void run_timemix_kernel(timemix_layout::globals g) {
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<timemix_kernel_template>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        mem_size);
    
    prototype::lcf::kernel<timemix_kernel_template>
        <<<g.grid(), g.block(), mem_size>>>(g);
}

// ============================================================================
// PYBIND11 BINDINGS
// ============================================================================

PYBIND11_MODULE(timemix_kernel, m) {
    m.doc() = "ThunderKittens TimeMix kernel for RWKV-style time mixing";
    
    py::bind_function<run_timemix_kernel>(
        m, "timemix",
        &timemix_layout::globals::x,
        &timemix_layout::globals::x_prev,
        &timemix_layout::globals::x_r,
        &timemix_layout::globals::x_w,
        &timemix_layout::globals::x_k,
        &timemix_layout::globals::x_v,
        &timemix_layout::globals::x_a,
        &timemix_layout::globals::x_g,
        &timemix_layout::globals::xr_out,
        &timemix_layout::globals::xw_out,
        &timemix_layout::globals::xk_out,
        &timemix_layout::globals::xv_out,
        &timemix_layout::globals::xa_out,
        &timemix_layout::globals::xg_out
    );
}