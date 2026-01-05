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
            return dim3(batch * T);  // One block per (batch, timestep)
        }
        dim3 block() { 
            return dim3(256);  // 8 warps, plenty of threads
        }
    };
};

__global__ void timemix_kernel(const __grid_constant__ timemix_layout::globals g) {
    using base_tile = timemix_layout::base_tile;
    
    // Shared memory for computation
    __shared__ base_tile smem_x;
    __shared__ base_tile smem_x_prev;
    __shared__ base_tile smem_xx;
    __shared__ base_tile smem_weight;
    __shared__ base_tile smem_result;
    
    int T = g.x.cols();
    int total_idx = blockIdx.x;
    
    int batch_idx = total_idx / T;
    int t = total_idx % T;
    
    int tid = threadIdx.x;
    constexpr int num_elements = 32 * 32;  // Elements in 32x32 tile
    
    // Load x and x_prev
    load(smem_x, g.x, {batch_idx, t, 0, 0});
    load(smem_x_prev, g.x_prev, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Compute xx = x_prev - x (elementwise, all threads participate)
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        smem_xx(row, col) = smem_x_prev(row, col) - smem_x(row, col);
    }
    __syncthreads();
    
    // Process all 6 outputs sequentially
    // Each output: result = x + xx * weight
    
    // Output 0: xr
    load(smem_weight, g.x_r, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xr_out, smem_result, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Output 1: xw
    load(smem_weight, g.x_w, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xw_out, smem_result, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Output 2: xk
    load(smem_weight, g.x_k, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xk_out, smem_result, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Output 3: xv
    load(smem_weight, g.x_v, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xv_out, smem_result, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Output 4: xa
    load(smem_weight, g.x_a, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xa_out, smem_result, {batch_idx, t, 0, 0});
    __syncthreads();
    
    // Output 5: xg
    load(smem_weight, g.x_g, {0, 0, 0, 0});
    __syncthreads();
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;
        float xx_val = __bfloat162float(smem_xx(row, col));
        float weight_val = __bfloat162float(smem_weight(row, col));
        float x_val = __bfloat162float(smem_x(row, col));
        smem_result(row, col) = __float2bfloat16(x_val + xx_val * weight_val);
    }
    __syncthreads();
    store(g.xg_out, smem_result, {batch_idx, t, 0, 0});
}

void run_timemix_kernel(timemix_layout::globals g) {
    timemix_kernel<<<g.grid(), g.block()>>>(g);
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