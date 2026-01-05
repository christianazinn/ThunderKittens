# test_timemix.py
import torch
import torch.nn.functional as F
import timemix_kernel  # Our compiled module
import numpy as np

def timemix_torch(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g):
    """
    Pure PyTorch reference implementation
    
    Args:
        x: [batch, T, 1024] input sequence
        x_prev: [batch, T, 1024] previous states
        x_r, x_w, x_k, x_v, x_a, x_g: [1024] weight vectors
    
    Returns:
        Tuple of (xr, xw, xk, xv, xa, xg), each [batch, T, 1024]
    """
    # Compute temporal difference
    xx = x_prev - x  # [batch, T, 1024]
    
    # Broadcast weights and compute weighted combinations
    # xx * x_r has shape [batch, T, 1024] (broadcasting x_r)
    xr = x + xx * x_r.unsqueeze(0).unsqueeze(0)
    xw = x + xx * x_w.unsqueeze(0).unsqueeze(0)
    xk = x + xx * x_k.unsqueeze(0).unsqueeze(0)
    xv = x + xx * x_v.unsqueeze(0).unsqueeze(0)
    xa = x + xx * x_a.unsqueeze(0).unsqueeze(0)
    xg = x + xx * x_g.unsqueeze(0).unsqueeze(0)
    
    return xr, xw, xk, xv, xa, xg

def test_timemix_kernel():
    """Test the ThunderKittens kernel against PyTorch reference"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch = 1
    T = 16  # sequence length
    D = 1024  # hidden dimension
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    print(f"Testing TimeMix kernel with batch={batch}, T={T}, D={D}")
    print("="*70)
    
    # Create random input tensors
    x = torch.randn(batch, T, D, device=device, dtype=dtype)
    x_prev = torch.randn(batch, T, D, device=device, dtype=dtype)
    
    # Weight vectors
    x_r = torch.randn(D, device=device, dtype=dtype)
    x_w = torch.randn(D, device=device, dtype=dtype)
    x_k = torch.randn(D, device=device, dtype=dtype)
    x_v = torch.randn(D, device=device, dtype=dtype)
    x_a = torch.randn(D, device=device, dtype=dtype)
    x_g = torch.randn(D, device=device, dtype=dtype)
    
    # Reshape for kernel (1024 -> 32x32)
    # Kernel expects [batch, T, 32, 32]
    x_kernel = x.view(batch, T, 32, 32).contiguous()
    x_prev_kernel = x_prev.view(batch, T, 32, 32).contiguous()
    x_r_kernel = x_r.view(1, 1, 32, 32).contiguous()
    x_w_kernel = x_w.view(1, 1, 32, 32).contiguous()
    x_k_kernel = x_k.view(1, 1, 32, 32).contiguous()
    x_v_kernel = x_v.view(1, 1, 32, 32).contiguous()
    x_a_kernel = x_a.view(1, 1, 32, 32).contiguous()
    x_g_kernel = x_g.view(1, 1, 32, 32).contiguous()
    
    # Allocate output tensors for kernel
    xr_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xw_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xk_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xv_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xa_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xg_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    
    # Run PyTorch reference
    print("\n1. Running PyTorch reference implementation...")
    xr_torch, xw_torch, xk_torch, xv_torch, xa_torch, xg_torch = \
        timemix_torch(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g)
    print("   ✓ PyTorch complete")
    
    # Run ThunderKittens kernel
    print("\n2. Running ThunderKittens kernel...")
    timemix_kernel.timemix(
        x_kernel,
        x_prev_kernel,
        x_r_kernel,
        x_w_kernel,
        x_k_kernel,
        x_v_kernel,
        x_a_kernel,
        x_g_kernel,
        xr_out_kernel,
        xw_out_kernel,
        xk_out_kernel,
        xv_out_kernel,
        xa_out_kernel,
        xg_out_kernel
    )
    torch.cuda.synchronize()
    print("   ✓ ThunderKittens kernel complete")
    
    # Reshape kernel outputs back to [batch, T, 1024]
    xr_kernel_flat = xr_out_kernel.view(batch, T, D)
    xw_kernel_flat = xw_out_kernel.view(batch, T, D)
    xk_kernel_flat = xk_out_kernel.view(batch, T, D)
    xv_kernel_flat = xv_out_kernel.view(batch, T, D)
    xa_kernel_flat = xa_out_kernel.view(batch, T, D)
    xg_kernel_flat = xg_out_kernel.view(batch, T, D)
    
    # Compare results
    print("\n3. Comparing results...")
    print("="*70)
    
    outputs = [
        ("xr", xr_torch, xr_kernel_flat),
        ("xw", xw_torch, xw_kernel_flat),
        ("xk", xk_torch, xk_kernel_flat),
        ("xv", xv_torch, xv_kernel_flat),
        ("xa", xa_torch, xa_kernel_flat),
        ("xg", xg_torch, xg_kernel_flat),
    ]
    
    all_passed = True
    for name, torch_out, kernel_out in outputs:
        # Compute error metrics
        abs_diff = torch.abs(torch_out - kernel_out)
        max_abs_error = abs_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        
        # Relative error
        rel_diff = abs_diff / (torch.abs(torch_out) + 1e-8)
        max_rel_error = rel_diff.max().item()
        mean_rel_error = rel_diff.mean().item()
        
        # Check if results match (accounting for bf16 precision)
        # bf16 has ~3 decimal digits of precision
        matches = torch.allclose(torch_out, kernel_out, rtol=1e-2, atol=1e-2)
        
        status = "✓ PASS" if matches else "✗ FAIL"
        all_passed = all_passed and matches
        
        print(f"\n{name}:")
        print(f"  Status:          {status}")
        print(f"  Max abs error:   {max_abs_error:.6f}")
        print(f"  Mean abs error:  {mean_abs_error:.6f}")
        print(f"  Max rel error:   {max_rel_error:.6f}")
        print(f"  Mean rel error:  {mean_rel_error:.6f}")
        
        if not matches:
            # Show some examples of mismatches
            mismatch_indices = torch.where(abs_diff > 1e-2)
            n_mismatches = len(mismatch_indices[0])
            print(f"  Mismatches:      {n_mismatches}/{torch_out.numel()}")
            
            if n_mismatches > 0:
                # Show first few mismatches
                for i in range(min(5, n_mismatches)):
                    idx = tuple(idx[i].item() for idx in mismatch_indices)
                    torch_val = torch_out[idx].item()
                    kernel_val = kernel_out[idx].item()
                    print(f"    [{idx}]: torch={torch_val:.6f}, kernel={kernel_val:.6f}")
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("❌ Some tests FAILED")
    print("="*70)
    
    return all_passed

def benchmark_timemix():
    """Benchmark ThunderKittens kernel vs PyTorch"""
    
    import time
    
    batch = 1
    T = 32
    D = 1024
    num_warmup = 2
    num_iters = 10
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    print(f"\nBenchmarking with batch={batch}, T={T}, D={D}")
    print("="*70)
    
    # Create inputs
    x = torch.randn(batch, T, D, device=device, dtype=dtype)
    x_prev = torch.randn(batch, T, D, device=device, dtype=dtype)
    x_r = torch.randn(D, device=device, dtype=dtype)
    x_w = torch.randn(D, device=device, dtype=dtype)
    x_k = torch.randn(D, device=device, dtype=dtype)
    x_v = torch.randn(D, device=device, dtype=dtype)
    x_a = torch.randn(D, device=device, dtype=dtype)
    x_g = torch.randn(D, device=device, dtype=dtype)
    
    # Prepare kernel inputs
    x_kernel = x.view(batch, T, 32, 32).contiguous()
    x_prev_kernel = x_prev.view(batch, T, 32, 32).contiguous()
    x_r_kernel = x_r.view(1, 1, 32, 32).contiguous()
    x_w_kernel = x_w.view(1, 1, 32, 32).contiguous()
    x_k_kernel = x_k.view(1, 1, 32, 32).contiguous()
    x_v_kernel = x_v.view(1, 1, 32, 32).contiguous()
    x_a_kernel = x_a.view(1, 1, 32, 32).contiguous()
    x_g_kernel = x_g.view(1, 1, 32, 32).contiguous()
    
    xr_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xw_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xk_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xv_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xa_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    xg_out_kernel = torch.zeros(batch, T, 32, 32, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        timemix_torch(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iters):
        timemix_torch(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Warmup kernel
    for _ in range(num_warmup):
        timemix_kernel.timemix_prefill(
            x_kernel, x_prev_kernel,
            x_r_kernel, x_w_kernel, x_k_kernel, x_v_kernel, x_a_kernel, x_g_kernel,
            xr_out_kernel, xw_out_kernel, xk_out_kernel, 
            xv_out_kernel, xa_out_kernel, xg_out_kernel
        )
    torch.cuda.synchronize()
    
    # Benchmark kernel
    start = time.time()
    for _ in range(num_iters):
        timemix_kernel.timemix_prefill(
            x_kernel, x_prev_kernel,
            x_r_kernel, x_w_kernel, x_k_kernel, x_v_kernel, x_a_kernel, x_g_kernel,
            xr_out_kernel, xw_out_kernel, xk_out_kernel,
            xv_out_kernel, xa_out_kernel, xg_out_kernel
        )
    torch.cuda.synchronize()
    kernel_time = (time.time() - start) / num_iters * 1000  # ms
    
    speedup = pytorch_time / kernel_time
    
    print(f"PyTorch time:        {pytorch_time:.4f} ms")
    print(f"ThunderKittens time: {kernel_time:.4f} ms")
    print(f"Speedup:             {speedup:.2f}x")
    print("="*70)

if __name__ == "__main__":
    # Run correctness test
    test_passed = test_timemix_kernel()
    
    # Run benchmark if test passed
    # if test_passed:
    #     benchmark_timemix()