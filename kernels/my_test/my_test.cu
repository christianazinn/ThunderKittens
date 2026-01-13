// timemix_binding.cu
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

// ============================================================================
// SIMPLE TIMEMIX KERNEL - One timestep per block
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
        glob\al_layout x_v;
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
            return dim3(batch * T);  // One block per (batch, timestep)
        }
        dim3 block() { 
            return dim3(256);  // 8 warps
        }
    };
};

__global__ void timemix_kernel(const __grid_constant__ timemix_layout::globals g) {
    using base_tile = timemix_layout::base_tile;
    
    // Shared memory for computation
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    base_tile &smem_x = al.allocate<base_tile>();
    base_tile &smem_x_prev = al.allocate<base_tile>();
    base_tile &smem_xx = al.allocate<base_tile>();
    base_tile &smem_weight = al.allocate<base_tile>();
    base_tile &smem_result = al.allocate<base_tile>();
    
    int T = g.x.cols();
    int total_idx = blockIdx.x;
    
    int batch_idx = total_idx / T;
    int t = total_idx % T;
    
    int tid = threadIdx.x;
    constexpr int num_elements = 32 * 32;
    
    // Load x and x_prev using TMA
    if (tid == 0) {
        tma::load_async(smem_x, g.x, {batch_idx, t, 0, 0});
        tma::load_async(smem_x_prev, g.x_prev, {batch_idx, t, 0, 0});
    }
    tma::arrive_and_wait();
    __syncthreads();
    
    // Compute xx = x_prev - x (elementwise)
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        smem_xx.data[i] = smem_x_prev.data[i] - smem_x.data[i];
    }
    __syncthreads();
    
    // Helper lambda for computing output
    auto compute_output = [&](auto& weight_global, auto& output_global) {
        // Load weight
        if (tid == 0) {
            tma::load_async(smem_weight, weight_global, {0, 0, 0, 0});
        }
        tma::arrive_and_wait();
        __syncthreads();
        
        // Compute result = x + xx * weight
        for (int i = tid; i < num_elements; i += blockDim.x) {
            float xx_val = __bfloat162float(smem_xx.data[i]);
            float weight_val = __bfloat162float(smem_weight.data[i]);
            float x_val = __bfloat162float(smem_x.data[i]);
            smem_result.data[i] = __float2bfloat16(x_val + xx_val * weight_val);
        }
        __syncthreads();
        
        // Store result
        if (tid == 0) {
            tma::store_async(output_global, smem_result, {batch_idx, t, 0, 0});
            tma::store_async_wait();
        }
        __syncthreads();
    };
    
    // Process all 6 outputs
    compute_output(g.x_r, g.xr_out);
    compute_output(g.x_w, g.xw_out);
    compute_output(g.x_k, g.xk_out);
    compute_output(g.x_v, g.xv_out);
    compute_output(g.x_a, g.xa_out);
    compute_output(g.x_g, g.xg_out);
}

void run_timemix_kernel(timemix_layout::globals g) {
    unsigned long mem_size = 5 * sizeof(st_bf<32, 32>) + 1024;  // 5 tiles + extra
    cudaFuncSetAttribute(timemix_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        mem_size);
    timemix_kernel<<<g.grid(), g.block(), mem_size>>>(g);
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