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
          sum += d_In[pixelIndex] * d_Filter[(i + filterSize / 2) * filterSize + j + filterSize / 2];
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
      printf("Processing job: %s -> %s -> %s\n", jobs[i]->source_name, getStrAlgoFilterByType(jobs[i]->processing_algo), jobs[i]->dest_name);

      cudaMemcpy(d_Filter, getAlgoFilterByType(jobs[i]->processing_algo), 25 * sizeof(float), cudaMemcpyHostToDevice);

      dim3 dimBlock(16, 16);
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
