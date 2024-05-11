#include "header.h"

int count_lines(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening job file <%s>.\n", filename);
        exit(-1);
    }
    int lineCount = 0;
    char buffer[1024]; // Buffer to store each line
    while (fgets(buffer, sizeof(buffer), file) != NULL)
    {
        lineCount++;
    }
    fclose(file);
    return lineCount;
}

PROCESSING_JOB **prepare_jobs(char *filename)
{
    char input_filename[100], processing_algo[100], output_filename[100];
    int count = 0;
    int nb_jobs = count_lines(filename);
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening job file <%s>.\n", filename);
        exit(-1);
    }

    PROCESSING_JOB **processing_job = (PROCESSING_JOB **)malloc((nb_jobs + 1) * sizeof(PROCESSING_JOB *));

    while (fscanf(file, "%s %s %s", input_filename, processing_algo, output_filename) != EOF)
    {
        PNG_RAW *png_raw = read_png(input_filename);
        if (png_raw->pixel_size != 3)
        {
            printf("Error, png file <%s> must be on 3 Bytes per pixel (not %d)\n", input_filename, png_raw->pixel_size);
            exit(0);
        }
        processing_job[count] = (PROCESSING_JOB *)malloc(sizeof(PROCESSING_JOB));
        processing_job[count]->source_name = strdup(input_filename);
        processing_job[count]->dest_name = strdup(output_filename);
        processing_job[count]->width = png_raw->width;
        processing_job[count]->height = png_raw->height;
        processing_job[count]->info_ptr = png_raw->info_ptr;
        processing_job[count]->pixel_size = png_raw->pixel_size;
        processing_job[count]->source_raw = png_raw->buf;
        processing_job[count]->dest_raw = (png_byte *)malloc(png_raw->width * png_raw->height * png_raw->pixel_size * sizeof(png_byte));

        clear_buf(processing_job[count]->dest_raw, png_raw->height, png_raw->width);
        processing_job[count]->processing_algo = getAlgoByName(processing_algo);
        count++;
    }
    processing_job[count] = NULL;
    fclose(file);

    return processing_job;
}

void write_jobs_output_files(PROCESSING_JOB **jobs)
{
    int count = 0;
    while (jobs[count] != NULL)
    {
        PNG_RAW *png_raw = (PNG_RAW *)malloc(sizeof(PNG_RAW));
        png_raw->height = jobs[count]->height;
        png_raw->width = jobs[count]->width;
        png_raw->pixel_size = jobs[count]->pixel_size;
        png_raw->buf = jobs[count]->dest_raw;
        png_raw->info_ptr = jobs[count]->info_ptr;
        printf("Writing output of job: %s -> %s\n", jobs[count]->source_name, jobs[count]->dest_name);
        write_png(jobs[count]->dest_name, png_raw);
        count++;
    }
}