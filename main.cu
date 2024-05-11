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

    // add other processing cases to main
    long long start_cpu = timeInMilliseconds();
    execute_jobs_cpu(jobs);
    long long end_cpu = timeInMilliseconds();

    // write_jobs_output_files(jobs);

    long long start_gpu = timeInMilliseconds();
    execute_jobs_gpu(jobs);
    long long end_gpu = timeInMilliseconds();

    // write_jobs_output_files(jobs);

    printf("Execution Time of CPU processing part: %lld ms\n", end_cpu - start_cpu);
    printf("Execution Time of GPU processing part: %lld ms\n", end_gpu - start_gpu);
}