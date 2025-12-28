#include "kittens.cuh"      // Main ThunderKittens library (tile ops, memory management)
#include "prototype.cuh"     // Template framework for producer-consumer patterns
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;  // Load-Compute-Finish template namespace

template <int T_BLOCK> struct op_layout {
    using  base_tile      = st_bf<32, 32>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout X, X_PREV,
                                          X_R, X_W, X_K, X_V, X_A, X_G,
                                          XR, XW, XK, XV, XA, XG; };
    struct input_block    { base_tile x, x_prev[T_BLOCK], x_r, x_w, x_k, x_v, x_a, x_g; };
    struct finish_block   { base_tile xr[T_BLOCK], xw[T_BLOCK], xk[T_BLOCK], xv[T_BLOCK], xa[T_BLOCK], xg[T_BLOCK]; };
    struct common_state   { int2 coord; };  // TODO: what?
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };  // TODO: what?
    struct scratch
};

template <int _T_BLOCK=1> struct op_template {
    static constexpr int T_BLOCK = _T_BLOCK;  // this is the number of Ts to process per {thread block}
    using layout    = op_layout<T_BLOCK>;

    // TODO: what?
    static constexpr int NUM_CONSUMER_WARPS=T_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    
    // TODO: what the fuck?
    template<bool PERSISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERSISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    
    // runs once per thread block
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }

    // load data duh
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                // dst, src, coord, arrived
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    // compute and store
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


PYBIND11_MODULE(my_test, m) {
    m.doc() = "example_bind python module";
    py::bind_kernel<copy_kernel>(m, "copy_kernel", &globals::in, &globals::out);
    py::bind_function<run_copy_kernel>(m, "wrapped_copy_kernel", &globals::in, &globals::out);
}
