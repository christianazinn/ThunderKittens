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
        global_layout x;          // [batch, T, 32, 32]
        global_layout x_prev;     // [batch, T, 32, 32]
        global_layout x_r;        // [1, 1, 32, 32]
        global_layout x_w;        // [1, 1, 32, 32]
        global_layout x_k;        // [1, 1, 32, 32]
        global_layout x_v;        // [1, 1, 32, 32]
        global_layout x_a;        // [1, 1, 32, 32]
        global_layout x_g;        // [1, 1, 32, 32]
        global_layout xr_out;     // [batch, T, 32, 32]
        global_layout xw_out;
        global_layout xk_out;
        global_layout xv_out;
        global_layout xa_out;
        global_layout xg_out;
        
        // Grid/block configuration
        dim3 grid()  { 
            int batch = x.batch();   // Call as function
            int T = x.cols();        // Call as function
            return dim3(batch * T);
        }
        dim3 block() { return dim3(128); }
    };
    
    struct input_block {
        base_tile x;
        base_tile x_prev;
        base_tile weights[6];
    };
    
    struct scratch_block {
        base_tile xx;
        base_tile outputs_temp[6];  // Store outputs here during compute
    };
    
    struct finish_block {
        base_tile outputs[6];
    };
    
    struct common_state {
        int batch_idx;
        int timestep;
    };
    
    struct consumer_state {
        rt_bf<32, 32> x_reg;
        rt_bf<32, 32> xx_reg;
        rt_bf<32, 32> weight_reg;
        rt_bf<32, 32> result_reg;
    };
};

// ============================================================================
// KERNEL TEMPLATE
// ============================================================================

struct timemix_kernel_template {
    using layout = timemix_layout;
    
    static constexpr int NUM_CONSUMER_WARPS = 4;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int T = args.globals.x.cols();  // Call as function
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
            // Note: Can't load weights here - no access to input_block or barriers
            // Weights will be loaded in the first load() call
        }
        
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                int batch = args.common.batch_idx;
                int t = args.common.timestep;
                
                // On first iteration, also load the weight vectors
                if (args.iter == 0) {
                    // Expect: 2 data tiles (x, x_prev) + 6 weight tiles
                    tma::expect(args.inputs_arrived, args.input, 8);
                    
                    // Load x and x_prev
                    tma::load_async(args.input.x, args.globals.x,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                    tma::load_async(args.input.x_prev, args.globals.x_prev,
                                   {batch, t, 0, 0}, args.inputs_arrived);
                    
                    // Load all 6 weight vectors
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
                    // Subsequent iterations: just load x and x_prev
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
    // CONSUMER
    // ========================================================================
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<100>();
        }
        
        __device__ static void compute(consumer_compute_args<layout> args) {
            // Load tiles from shared memory to registers
            warpgroup::load(args.state.x_reg, args.input.x);
            warpgroup::load(args.state.xx_reg, args.input.x_prev);
            
            // Compute xx = x_prev - x
            warpgroup::sub(args.state.xx_reg, args.state.xx_reg, args.state.x_reg);
            
            // Store xx to scratch
            warpgroup::store(args.scratch.xx, args.state.xx_reg);
            
            // Distribute 6 outputs across 4 warps
            int warp_id = warpgroup::warpid();
            int start_output = warp_id * 2;
            int end_output = min(start_output + 2, 6);
            
            for (int i = start_output; i < end_output; i++) {
                // Load weight for this output
                warpgroup::load(args.state.weight_reg, args.input.weights[i]);
                
                // Compute: result = xx * weight
                warpgroup::mul(args.state.result_reg, 
                              args.state.xx_reg, 
                              args.state.weight_reg);
                
                // Compute: result = x + (xx * weight)
                warpgroup::add(args.state.result_reg,
                              args.state.x_reg,
                              args.state.result_reg);
                
                // Store to scratch (not finish - that's not accessible here)
                warpgroup::store(args.scratch.outputs_temp[i], args.state.result_reg);
            }
            
            warpgroup::sync();
            if (laneid() == 0) arrive(args.inputs_finished);
        }
        
        __device__ static void finish(consumer_finish_args<layout> args) {
            // Copy from scratch to finish block
            int warp_id = warpgroup::warpid();
            
            if (warp_id == 0) {
                // Copy all 6 outputs from scratch to finish
                for (int i = 0; i < 6; i++) {
                    copy(args.finish.outputs[i], args.scratch.outputs_temp[i]);
                }
            }
            
            warpgroup::sync();
            
            // Now store to global memory
            int batch = args.common.batch_idx;
            int t = args.common.timestep;
            
            if (warp_id == 0) {
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
            }
            
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