#include "header.h"

float sharpen_filter[] = {
    -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f,
    -1 / 8.0f, 2 / 8.0f, 2 / 8.0f, 2 / 8.0f, -1 / 8.0f,
    -1 / 8.0f, 2 / 8.0f, 8 / 8.0f, 2 / 8.0f, -1 / 8.0f,
    -1 / 8.0f, 2 / 8.0f, 2 / 8.0f, 2 / 8.0f, -1 / 8.0f,
    -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f, -1 / 8.0f};

float box_blur_filter[] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

float edge_detect_filter[] = {
    -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
    -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
    -1 / 24.0f, -1 / 24.0f, 24 / 24.0f, -1 / 24.0f, -1 / 24.0f,
    -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f,
    -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f, -1 / 24.0f};

PROCESSING_ALGO getAlgoByName(char *algo_name)
{
    if (strcmp(algo_name, "sharpen") == 0)
        return SHARPEN;
    else if (strcmp(algo_name, "edge") == 0)
        return EDGE;
    else if (strcmp(algo_name, "blur") == 0)
        return BLUR;
    else
    {
        printf("Unknown Processing algorithm name <%s>\n", algo_name);
        exit(-1);
    }
}

float *getAlgoFilterByType(PROCESSING_ALGO algo)
{
    switch (algo)
    {
    case SHARPEN:
        return sharpen_filter;
    case EDGE:
        return edge_detect_filter;
    case BLUR:
        return box_blur_filter;
    default:
        return NULL;
    }
}

char *getStrAlgoFilterByType(PROCESSING_ALGO algo)
{
    switch (algo)
    {
    case SHARPEN:
        return strdup("sharpen");
    case EDGE:
        return strdup("edge");
    case BLUR:
        return strdup("blur");
    default:
        return strdup("unknown");
    }
}