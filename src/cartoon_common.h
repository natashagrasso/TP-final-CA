#pragma once

#include <vector>

#include "bmp.h"

struct PixelStats {
    int min = 255;
    int max = 0;
};

std::vector<float> makeBoxKernel(int size);
std::vector<float> makeSobelXKernel();
std::vector<float> makeSobelYKernel();

Image blurSequential(const Image& image, int kernelSize);
Image blurOpenMP(const Image& image, int kernelSize);
Image blurMPI(const Image& image, int kernelSize, int rank, int worldSize);
Image detectEdgesSequential(const Image& image, int kernelSize, int threshold);
Image detectEdgesOpenMP(const Image& image, int kernelSize, int threshold);
Image detectEdgesMPI(const Image& image, int kernelSize, int threshold, int rank, int worldSize);
Image posterizeSequential(const Image& image, int levels);
Image posterizeOpenMP(const Image& image, int levels);
Image posterizeMPI(const Image& image, int levels, int rank, int worldSize);
Image cartoonizeSequential(const Image& image, int kernelSize, int levels, int threshold);
Image cartoonizeOpenMP(const Image& image, int kernelSize, int levels, int threshold);
Image cartoonizeMPI(const Image& image, int kernelSize, int levels, int threshold, int rank, int worldSize);

float clampFloat(float value, float minValue, float maxValue);
uint8_t clampByte(int value);
