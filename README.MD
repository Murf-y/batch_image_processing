# Batch Image Processing with CUDA

## Usage

```bash
!nvcc main.cu cpu_processing.cu filters.cu job_loader.cu png_tools.cu -lpng -o cuda_app.out

!./cuda_app.out jobs_batch.txt
```

To profile:

```bash
!nvprof ./cuda_app.out jobs_batch.txt
```

## Performance

### CPU

```md
Execution Time of CPU processing part: 28214 ms

Execution Time of GPU processing part: 257 ms
```

- As we can see, the CPU processing part took 28214 ms, while the GPU processing part took 257 ms. This is a significant improvement in performance.

- Speedup: 28214 / 257 = 109.8 times
