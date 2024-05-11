#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <sys/time.h>

#ifndef GLOBALS_H
#define GLOBALS_H
extern float sharpen_filter[];
extern float edge_detec_filter[];
extern float box_blur_filter[];
#endif

typedef enum
{
    SHARPEN,
    EDGE,
    BLUR,
    UNKNOWN
} PROCESSING_ALGO;

typedef struct
{
    int height;
    int width;
    int pixel_size;
    png_infop info_ptr;
    png_byte *buf;
} PNG_RAW;

typedef struct
{
    char *source_name;
    char *dest_name;
    int width;
    int height;
    int pixel_size;
    png_infop info_ptr;
    png_byte *source_raw;
    png_byte *dest_raw;
    PROCESSING_ALGO processing_algo;
} PROCESSING_JOB;

PNG_RAW *read_png(char *file_name);
void write_png(char *file_name, PNG_RAW *png_raw);
void clear_buf(png_byte *buf, int height, int width);
PROCESSING_JOB **prepare_jobs(char *filename);
void write_jobs_output_files(PROCESSING_JOB **jobs);
void execute_jobs_cpu(PROCESSING_JOB **jobs); // jobs NULL terminated
void execute_jobs_gpu(PROCESSING_JOB **jobs); // jobs NULL terminated
PROCESSING_ALGO getAlgoByName(char *algo_name);
float *getAlgoFilterByType(PROCESSING_ALGO algo);
char *getStrAlgoFilterByType(PROCESSING_ALGO algo);