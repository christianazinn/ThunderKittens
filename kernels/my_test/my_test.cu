// timemix_binding.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// ============================================================================
// SIMPLIFIED TIMEMIX KERNEL (Non-templated for easier binding)
// ============================================================================

// Layout for the timemix kernel
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
            int batch = x.batch;
            int T = x.cols;
            return dim3(batch * T);  // One block per (batch, timestep)
        }
        dim3 block() { return dim3(128); }  // 128 threads = 4 warps = 1 warpgroup
    };
    
    struct input_block {
        base_tile x;
        base_tile x_prev;
        base_tile weights[6];
    };
    
    struct scratch_block {
        base_tile xx;
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

// Simple template for single timestep per block
struct timemix_kernel_template {
    using layout = timemix_layout;
    
    static constexpr int NUM_CONSUMER_WARPS = 4;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int T = args.globals.x.cols;
        int total_task = blockIdx.x;
        
        args.common.batch_idx = total_task / T;
        args.common.timestep = total_task % T;
        args.num_iters = 1;
    }
    
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
            
            // Load weights once during setup
            if (warpgroup::warpid() == 0 && args.iter == 0) {
                tma::expect_bytes(args.inputs_arrived, 
                                 sizeof(typename layout::base_tile) * 6);
                
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
            }
        }
        
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                int batch = args.common.batch_idx;
                int t = args.common.timestep;
                
                tma::expect(args.inputs_arrived, args.input, 2);
                
                tma::load_async(args.input.x, args.globals.x,
                               {batch, t, 0, 0}, args.inputs_arrived);
                tma::load_async(args.input.x_prev, args.globals.x_prev,
                               {batch, t, 0, 0}, args.inputs_arrived);
            }
        }
    };
    
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<100>();
        }
        
        __device__ static void compute(consumer_compute_args<layout> args) {
            // Load tiles into registers
            warpgroup::load(args.state.x_reg, args.input.x);
            warpgroup::load(args.state.xx_reg, args.input.x_prev);
            
            // Compute xx = x_prev - x
            warpgroup::sub(args.state.xx_reg, args.state.xx_reg, args.state.x_reg);
            warpgroup::store(args.scratch.xx, args.state.xx_reg);
            
            // Distribute 6 outputs across 4 warps
            // Warp 0: outputs 0, 1
            // Warp 1: outputs 2, 3
            // Warp 2: outputs 4, 5
            // Warp 3: idle
            int warp_id = warpgroup::warpid();
            int start_output = warp_id * 2;
            int end_output = min(start_output + 2, 6);
            
            for (int i = start_output; i < end_output; i++) {
                warpgroup::load(args.state.weight_reg, args.input.weights[i]);
                
                // result = xx * weight
                warpgroup::mul(args.state.result_reg, 
                              args.state.xx_reg, 
                              args.state.weight_reg);
                
                // result = x + (xx * weight)
                warpgroup::add(args.state.result_reg,
                              args.state.x_reg,
                              args.state.result_reg);
                
                warpgroup::store(args.finish.outputs[i], args.state.result_reg);
            }
            
            warpgroup::sync();
            if (laneid() == 0) arrive(args.inputs_finished);
        }
        
        __device__ static void finish(consumer_finish_args<layout> args) {
            int batch = args.common.batch_idx;
            int t = args.common.timestep;
            
            if (warpgroup::warpid() == 0) {
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

// Wrapper function for easier binding
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
    
    // Bind the kernel
    // Lists all the input and output tensors in order
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