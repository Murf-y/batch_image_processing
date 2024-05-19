#include "header.h"

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

int main(int argc, char **argv)
{
    PROCESSING_JOB **jobs = prepare_jobs(argv[1]);

    // Timing and executing CPU jobs
    long long start_cpu = timeInMilliseconds();
    execute_jobs_cpu(jobs);
    long long end_cpu = timeInMilliseconds();

    // Timing and executing GPU jobs (original version)
    long long start_gpu = timeInMilliseconds();
    execute_jobs_gpu(jobs);
    long long end_gpu = timeInMilliseconds();

    // // Timing and executing GPU jobs (version 2 with image tiling)
    long long start_gpu_v2 = timeInMilliseconds();
    execute_jobs_gpu_v2(jobs);
    long long end_gpu_v2 = timeInMilliseconds();

    printf("Execution Time of CPU processing part: %lld ms\n", end_cpu - start_cpu);
    printf("Execution Time of GPU processing part (Original): %lld ms\n", end_gpu - start_gpu);
    printf("Execution Time of GPU processing part (Version 2 with Tiling): %lld ms\n", end_gpu_v2 - start_gpu_v2);

    // Optionally, write output files for each job set to compare outputs visually or via other means
    write_jobs_output_files(jobs); // You could uncomment and modify this to differentiate outputs per method if needed
}