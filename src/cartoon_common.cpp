#include "cartoon_common.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace {

inline int clampToInt(int value, int minValue, int maxValue) {
    return std::max(minValue, std::min(maxValue, value));
}

std::vector<uint8_t> toGray(const Image& image) {
    std::vector<uint8_t> gray(static_cast<size_t>(image.width * image.height));
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const RGB& pixel = image.at(x, y);
            int value = (pixel.r + pixel.g + pixel.b) / 3;
            gray[static_cast<size_t>(y * image.width + x)] = static_cast<uint8_t>(value);
        }
    }
    return gray;
}

Image applyKernel(const Image& image, const std::vector<float>& kernel, int kernelSize) {
    const int radius = kernelSize / 2;
    Image result = makeImage(image.width, image.height);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float sumR = 0.0f;
            float sumG = 0.0f;
            float sumB = 0.0f;

            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int sampleX = clampToInt(x + (kx - radius), 0, image.width - 1);
                    int sampleY = clampToInt(y + (ky - radius), 0, image.height - 1);
                    const RGB& pixel = image.at(sampleX, sampleY);
                    const float weight = kernel[static_cast<size_t>(ky * kernelSize + kx)];
                    sumR += static_cast<float>(pixel.r) * weight;
                    sumG += static_cast<float>(pixel.g) * weight;
                    sumB += static_cast<float>(pixel.b) * weight;
                }
            }

            result.at(x, y) = RGB(clampByte(static_cast<int>(std::lround(sumR))),
                                  clampByte(static_cast<int>(std::lround(sumG))),
                                  clampByte(static_cast<int>(std::lround(sumB))));
        }
    }

    return result;
}

Image applyKernelOpenMP(const Image& image, const std::vector<float>& kernel, int kernelSize) {
    const int radius = kernelSize / 2;
    Image result = makeImage(image.width, image.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float sumR = 0.0f;
            float sumG = 0.0f;
            float sumB = 0.0f;

            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int sampleX = clampToInt(x + (kx - radius), 0, image.width - 1);
                    int sampleY = clampToInt(y + (ky - radius), 0, image.height - 1);
                    const RGB& pixel = image.at(sampleX, sampleY);
                    const float weight = kernel[static_cast<size_t>(ky * kernelSize + kx)];
                    sumR += static_cast<float>(pixel.r) * weight;
                    sumG += static_cast<float>(pixel.g) * weight;
                    sumB += static_cast<float>(pixel.b) * weight;
                }
            }

            result.at(x, y) = RGB(clampByte(static_cast<int>(std::lround(sumR))),
                                  clampByte(static_cast<int>(std::lround(sumG))),
                                  clampByte(static_cast<int>(std::lround(sumB))));
        }
    }

    return result;
}

Image detectGradientEdges(const Image& image, int threshold) {
    const std::vector<float> sobelX = makeSobelXKernel();
    const std::vector<float> sobelY = makeSobelYKernel();
    Image edges = makeImage(image.width, image.height);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int gx = 0;
            int gy = 0;

            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int sampleX = clampToInt(x + (kx - 1), 0, image.width - 1);
                    int sampleY = clampToInt(y + (ky - 1), 0, image.height - 1);
                    const RGB& pixel = image.at(sampleX, sampleY);
                    int gray = (pixel.r + pixel.g + pixel.b) / 3;
                    float weightX = sobelX[static_cast<size_t>(ky * 3 + kx)];
                    float weightY = sobelY[static_cast<size_t>(ky * 3 + kx)];
                    gx += static_cast<int>(gray * weightX);
                    gy += static_cast<int>(gray * weightY);
                }
            }

            int magnitude = std::abs(gx) + std::abs(gy);
            edges.at(x, y) = magnitude > threshold ? RGB(255, 255, 255) : RGB(0, 0, 0);
        }
    }

    return edges;
}

Image detectGradientEdgesOpenMP(const Image& image, int threshold) {
    const std::vector<float> sobelX = makeSobelXKernel();
    const std::vector<float> sobelY = makeSobelYKernel();
    Image edges = makeImage(image.width, image.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int gx = 0;
            int gy = 0;

            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int sampleX = clampToInt(x + (kx - 1), 0, image.width - 1);
                    int sampleY = clampToInt(y + (ky - 1), 0, image.height - 1);
                    const RGB& pixel = image.at(sampleX, sampleY);
                    int gray = (pixel.r + pixel.g + pixel.b) / 3;
                    float weightX = sobelX[static_cast<size_t>(ky * 3 + kx)];
                    float weightY = sobelY[static_cast<size_t>(ky * 3 + kx)];
                    gx += static_cast<int>(gray * weightX);
                    gy += static_cast<int>(gray * weightY);
                }
            }

            int magnitude = std::abs(gx) + std::abs(gy);
            edges.at(x, y) = magnitude > threshold ? RGB(255, 255, 255) : RGB(0, 0, 0);
        }
    }

    return edges;
}

} // namespace

float clampFloat(float value, float minValue, float maxValue) {
    return std::max(minValue, std::min(maxValue, value));
}

uint8_t clampByte(int value) {
    return static_cast<uint8_t>(clampToInt(value, 0, 255));
}

std::vector<float> makeBoxKernel(int size) {
    if (size <= 0 || size % 2 == 0) {
        throw std::invalid_argument("El tamaño del kernel debe ser impar y mayor que cero.");
    }

    std::vector<float> kernel(static_cast<size_t>(size * size), 1.0f / static_cast<float>(size * size));
    return kernel;
}

std::vector<float> makeSobelXKernel() {
    return {-1.0f, 0.0f, 1.0f,
            -2.0f, 0.0f, 2.0f,
            -1.0f, 0.0f, 1.0f};
}

std::vector<float> makeSobelYKernel() {
    return {-1.0f, -2.0f, -1.0f,
             0.0f,  0.0f,  0.0f,
             1.0f,  2.0f,  1.0f};
}

Image blurSequential(const Image& image, int kernelSize) {
    return applyKernel(image, makeBoxKernel(kernelSize), kernelSize);
}

Image blurOpenMP(const Image& image, int kernelSize) {
    return applyKernelOpenMP(image, makeBoxKernel(kernelSize), kernelSize);
}

Image blurMPI(const Image& image, int kernelSize, int, int) {
    return blurSequential(image, kernelSize);
}

Image detectEdgesSequential(const Image& image, int kernelSize, int threshold) {
    Image blurred = blurSequential(image, kernelSize);
    return detectGradientEdges(blurred, threshold);
}

Image detectEdgesOpenMP(const Image& image, int kernelSize, int threshold) {
    Image blurred = blurOpenMP(image, kernelSize);
    return detectGradientEdgesOpenMP(blurred, threshold);
}

Image detectEdgesMPI(const Image& image, int kernelSize, int threshold, int, int) {
    return detectEdgesSequential(image, kernelSize, threshold);
}

Image posterizeSequential(const Image& image, int levels) {
    if (levels <= 1) {
        return image;
    }

    Image result = makeImage(image.width, image.height);
    const int scale = std::max(1, levels - 1);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const RGB& pixel = image.at(x, y);
            auto quantize = [&](uint8_t value) {
                int normalized = static_cast<int>(std::lround((static_cast<float>(value) / 255.0f) * scale));
                normalized = std::max(0, std::min(scale, normalized));
                return static_cast<uint8_t>(std::lround((normalized * 255.0f) / scale));
            };

            result.at(x, y) = RGB(quantize(pixel.r), quantize(pixel.g), quantize(pixel.b));
        }
    }

    return result;
}

Image posterizeOpenMP(const Image& image, int levels) {
    if (levels <= 1) {
        return image;
    }

    Image result = makeImage(image.width, image.height);
    const int scale = std::max(1, levels - 1);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const RGB& pixel = image.at(x, y);
            auto quantize = [&](uint8_t value) {
                int normalized = static_cast<int>(std::lround((static_cast<float>(value) / 255.0f) * scale));
                normalized = std::max(0, std::min(scale, normalized));
                return static_cast<uint8_t>(std::lround((normalized * 255.0f) / scale));
            };

            result.at(x, y) = RGB(quantize(pixel.r), quantize(pixel.g), quantize(pixel.b));
        }
    }

    return result;
}

Image posterizeMPI(const Image& image, int levels, int, int) {
    return posterizeSequential(image, levels);
}

Image cartoonizeSequential(const Image& image, int kernelSize, int levels, int threshold) {
    Image posterized = posterizeSequential(image, levels);
    Image edges = detectEdgesSequential(image, kernelSize, threshold);
    Image output = makeImage(image.width, image.height);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const RGB& p = posterized.at(x, y);
            const RGB& edge = edges.at(x, y);
            if (edge.r > 10 || edge.g > 10 || edge.b > 10) {
                output.at(x, y) = RGB(0, 0, 0);
            } else {
                output.at(x, y) = p;
            }
        }
    }

    return output;
}

Image cartoonizeOpenMP(const Image& image, int kernelSize, int levels, int threshold) {
    Image posterized = posterizeOpenMP(image, levels);
    Image edges = detectEdgesOpenMP(image, kernelSize, threshold);
    Image output = makeImage(image.width, image.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const RGB& p = posterized.at(x, y);
            const RGB& edge = edges.at(x, y);
            if (edge.r > 10 || edge.g > 10 || edge.b > 10) {
                output.at(x, y) = RGB(0, 0, 0);
            } else {
                output.at(x, y) = p;
            }
        }
    }

    return output;
}

Image cartoonizeMPI(const Image& image, int kernelSize, int levels, int threshold, int, int) {
    return cartoonizeSequential(image, kernelSize, levels, threshold);
}
