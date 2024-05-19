#include "header.h"
#include <string.h>

// Comparison function for sorting jobs by source name
int compare_jobs(const void *a, const void *b)
{
  PROCESSING_JOB *jobA = *(PROCESSING_JOB **)a;
  PROCESSING_JOB *jobB = *(PROCESSING_JOB **)b;
  return strcmp(jobA->source_name, jobB->source_name);
}

void sort_jobs(PROCESSING_JOB **jobs)
{
  int count = 0;
  while (jobs[count] != NULL)
    count++; // Count how many jobs are in the array
  qsort(jobs, count, sizeof(PROCESSING_JOB *), compare_jobs);
}

void PictureHost_FILTER(png_byte *h_In, png_byte *h_Out, int h, int w, float *h_filt)
{
  float out;
  png_byte b;

  for (int Row = 2; Row < h - 2; Row++)
    for (int Col = 2; Col < w - 2; Col++)
    {
      for (int color = 0; color < 3; color++)
      {
        out = 0;
        for (int i = -2; i <= 2; i++)
          for (int j = -2; j <= 2; j++)
            out += h_filt[(i + 2) * 5 + j + 2] * h_In[((Row + i) * w + (Col + j)) * 3 + color];
        b = (png_byte)fminf(fmaxf(out, 0.0f), 255.0f);
        h_Out[(Row * w + Col) * 3 + color] = b;
      }
    }
}

void execute_jobs_cpu(PROCESSING_JOB **jobs)
{
  int count = 0;
  float *h_filter;
  while (jobs[count] != NULL)
  {
    printf("Processing job: %s -> %s -> %s\n", jobs[count]->source_name, getStrAlgoFilterByType(jobs[count]->processing_algo), jobs[count]->dest_name);

    h_filter = getAlgoFilterByType(jobs[count]->processing_algo);
    PictureHost_FILTER(jobs[count]->source_raw, jobs[count]->dest_raw,
                       jobs[count]->height, jobs[count]->width, h_filter);
    count++;
  }
}

__global__ void PictureDevice_FILTER(png_byte *d_In, png_byte *d_Out, int h, int w, float *d_Filter, int filterSize)
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  // Its better to use shared memory because it reduces the number of global memory accesses
  __shared__ float shared_Filter[25];

  // Load filter into shared memory
  if (threadIdx.x < filterSize && threadIdx.y < filterSize)
    shared_Filter[threadIdx.y * filterSize + threadIdx.x] = d_Filter[threadIdx.y * filterSize + threadIdx.x];
  __syncthreads();

  if (Row >= filterSize / 2 && Row < h - filterSize / 2 && Col >= filterSize / 2 && Col < w - filterSize / 2)
  {
    for (int color = 0; color < 3; color++)
    {
      float sum = 0.0;
      for (int i = -filterSize / 2; i <= filterSize / 2; i++)
      {
        for (int j = -filterSize / 2; j <= filterSize / 2; j++)
        {
          int pixelIndex = ((Row + i) * w + (Col + j)) * 3 + color;
          sum += d_In[pixelIndex] * shared_Filter[(i + filterSize / 2) * filterSize + j + filterSize / 2];
        }
      }
      d_Out[(Row * w + Col) * 3 + color] = fminf(fmaxf(sum, 0.0f), 255.0f);
    }
  }
}
void execute_jobs_gpu(PROCESSING_JOB **jobs)
{
  png_byte *d_In, *d_Out;
  float *d_Filter;
  int currentImageHeight = 0, currentImageWidth = 0;

  cudaMalloc(&d_Filter, 25 * sizeof(float)); // Assuming all filters are 5x5
  sort_jobs(jobs);                           // Sort jobs by image before processing

  for (int i = 0; jobs[i] != NULL;)
  {
    // Check if we need to load new image data
    // This prevents us from loading the same image multiple times
    if (i == 0 || strcmp(jobs[i]->source_name, jobs[i - 1]->source_name) != 0)
    {
      if (i != 0)
        cudaFree(d_In);
      cudaMalloc(&d_In, jobs[i]->height * jobs[i]->width * 3 * sizeof(png_byte));
      cudaMemcpy(d_In, jobs[i]->source_raw, jobs[i]->height * jobs[i]->width * 3 * sizeof(png_byte), cudaMemcpyHostToDevice);
      currentImageHeight = jobs[i]->height;
      currentImageWidth = jobs[i]->width;
    }

    cudaMalloc(&d_Out, currentImageHeight * currentImageWidth * 3 * sizeof(png_byte));

    do
    {
      // printf("Processing job: %s -> %s -> %s\n", jobs[i]->source_name, getStrAlgoFilterByType(jobs[i]->processing_algo), jobs[i]->dest_name);

      cudaMemcpy(d_Filter, getAlgoFilterByType(jobs[i]->processing_algo), 25 * sizeof(float), cudaMemcpyHostToDevice);

      dim3 dimBlock(32, 32);
      dim3 dimGrid((currentImageWidth + dimBlock.x - 1) / dimBlock.x, (currentImageHeight + dimBlock.y - 1) / dimBlock.y);
      PictureDevice_FILTER<<<dimGrid, dimBlock>>>(d_In, d_Out, currentImageHeight, currentImageWidth, d_Filter, 5);

      cudaMemcpy(jobs[i]->dest_raw, d_Out, currentImageHeight * currentImageWidth * 3 * sizeof(png_byte), cudaMemcpyDeviceToHost);

      i++;
    } while (jobs[i] != NULL && strcmp(jobs[i]->source_name, jobs[i - 1]->source_name) == 0);

    cudaFree(d_Out);
  }

  cudaFree(d_In);
  cudaFree(d_Filter);
}

__global__ void PictureDevice_FILTER_v2(png_byte *d_In, png_byte *d_Out, int h, int w, float *d_Filter, int filterSize)
{
    int Col = blockIdx.x * (blockDim.x - filterSize + 1) + threadIdx.x;
    int Row = blockIdx.y * (blockDim.y - filterSize + 1) + threadIdx.y;

    __shared__ float shared_Filter[25];
    __shared__ png_byte shared_Image[(32 + 4) * (32 + 4) * 3]; // Adjust size based on maximum block size and filter overlap

    // Load filter into shared memory
    if (threadIdx.x < filterSize && threadIdx.y < filterSize) {
        shared_Filter[threadIdx.y * filterSize + threadIdx.x] = d_Filter[threadIdx.y * filterSize + threadIdx.x];
    }

    // Load image data into shared memory
    int sharedIndex = (threadIdx.y * (blockDim.x + filterSize - 1) + threadIdx.x) * 3;
    int globalIndex = ((Row + (filterSize / 2)) * w + (Col + (filterSize / 2))) * 3;
    if (Row + (filterSize / 2) < h && Col + (filterSize / 2) < w)
        for (int c = 0; c < 3; c++) {
            shared_Image[sharedIndex + c] = d_In[globalIndex + c];
        }

    __syncthreads();

    if (threadIdx.x >= filterSize / 2 && threadIdx.x < blockDim.x - filterSize / 2 && threadIdx.y >= filterSize / 2 && threadIdx.y < blockDim.y - filterSize / 2 && Row < h - filterSize / 2 && Col < w - filterSize / 2) {
        for (int color = 0; color < 3; color++) {
            float sum = 0.0;
            for (int i = -filterSize / 2; i <= filterSize / 2; i++) {
                for (int j = -filterSize / 2; j <= filterSize / 2; j++) {
                    int sharedPixelIndex = ((threadIdx.y + i) * (blockDim.x + filterSize - 1) + (threadIdx.x + j)) * 3 + color;
                    sum += shared_Image[sharedPixelIndex] * shared_Filter[(i + filterSize / 2) * filterSize + j + filterSize / 2];
                }
            }
            d_Out[(Row * w + Col) * 3 + color] = fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

void execute_jobs_gpu_v2(PROCESSING_JOB **jobs)
{
    png_byte *d_In, *d_Out;
    float *d_Filter;
    int currentImageHeight = 0, currentImageWidth = 0;

    cudaMalloc(&d_Filter, 25 * sizeof(float)); // Assuming all filters are 5x5
    sort_jobs(jobs);                           // Sort jobs by image before processing

    for (int i = 0; jobs[i] != NULL;)
    {
        if (i == 0 || strcmp(jobs[i]->source_name, jobs[i - 1]->source_name) != 0)
        {
            if (i != 0)
                cudaFree(d_In);
            cudaMalloc(&d_In, jobs[i]->height * jobs[i]->width * 3 * sizeof(png_byte));
            cudaMemcpy(d_In, jobs[i]->source_raw, jobs[i]->height * jobs[i]->width * 3 * sizeof(png_byte), cudaMemcpyHostToDevice);
            currentImageHeight = jobs[i]->height;
            currentImageWidth = jobs[i]->width;
        }

        cudaMalloc(&d_Out, currentImageHeight * currentImageWidth * 3 * sizeof(png_byte));

        do
        {
            cudaMemcpy(d_Filter, getAlgoFilterByType(jobs[i]->processing_algo), 25 * sizeof(float), cudaMemcpyHostToDevice);

            dim3 dimBlock(32, 32);
            dim3 dimGrid((currentImageWidth + dimBlock.x - 5) / (dimBlock.x - 4), (currentImageHeight + dimBlock.y - 5) / (dimBlock.y - 4));
            PictureDevice_FILTER_v2<<<dimGrid, dimBlock>>>(d_In, d_Out, currentImageHeight, currentImageWidth, d_Filter, 5);

            cudaMemcpy(jobs[i]->dest_raw, d_Out, currentImageHeight * currentImageWidth * 3 * sizeof(png_byte), cudaMemcpyDeviceToHost);

            i++;
        } while (jobs[i] != NULL && strcmp(jobs[i]->source_name, jobs[i - 1]->source_name) == 0);

        cudaFree(d_Out);
    }

    cudaFree(d_In);
    cudaFree(d_Filter);
}

