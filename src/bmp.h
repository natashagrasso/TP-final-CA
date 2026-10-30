#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct RGB {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    RGB(uint8_t red = 0, uint8_t green = 0, uint8_t blue = 0)
        : r(red), g(green), b(blue) {}
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<RGB> pixels;

    RGB& at(int x, int y);
    const RGB& at(int x, int y) const;
};

Image makeImage(int width, int height);
Image loadBMP(const std::string& path);
void saveBMP(const std::string& path, const Image& image);
Image generateTestImage(int width, int height);
