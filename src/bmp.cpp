#include "bmp.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

#pragma pack(push, 1)
struct BitmapFileHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
};

struct BitmapInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

}

RGB& Image::at(int x, int y) {
    return pixels[static_cast<size_t>(y * width + x)];
}

const RGB& Image::at(int x, int y) const {
    return pixels[static_cast<size_t>(y * width + x)];
}

Image makeImage(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(static_cast<size_t>(width * height), RGB(0, 0, 0));
    return img;
}

Image generateTestImage(int width, int height) {
    Image img = makeImage(width, height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dx = x - width / 2;
            int dy = y - height / 2;
            int dist = dx * dx + dy * dy;

            uint8_t r = 0;
            uint8_t g = 0;
            uint8_t b = 0;

            if (dist < (width * height) / 14) {
                r = static_cast<uint8_t>(std::min(255, 90 + (x * 3) % 135));
                g = static_cast<uint8_t>(std::min(255, 120 + (y * 2) % 120));
                b = static_cast<uint8_t>(std::min(255, 160 + ((x + y) * 2) % 90));
            }

            if ((x > width * 0.25 && x < width * 0.75) && (y > height * 0.25 && y < height * 0.75)) {
                r = static_cast<uint8_t>(std::min(255, 40 + (x * 2) % 180));
                g = static_cast<uint8_t>(std::min(255, 50 + (y * 2) % 180));
                b = static_cast<uint8_t>(std::min(255, 200 + (x + y) % 30));
            }

            if (x % 17 == 0 || y % 19 == 0) {
                r = 255;
                g = 180;
                b = 80;
            }

            img.at(x, y) = RGB(r, g, b);
        }
    }

    return img;
}

Image loadBMP(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir la imagen de entrada: " + path);
    }

    BitmapFileHeader fileHeader{};
    BitmapInfoHeader infoHeader{};

    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (fileHeader.type != 0x4D42) {
        throw std::runtime_error("El archivo no es un BMP válido: " + path);
    }
    if (infoHeader.bitCount != 24 || infoHeader.compression != 0) {
        throw std::runtime_error("Se requiere un BMP de 24 bits sin compresión: " + path);
    }

    int width = std::abs(infoHeader.width);
    int height = std::abs(infoHeader.height);

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("La imagen BMP tiene dimensiones inválidas: " + path);
    }

    size_t rowStride = static_cast<size_t>((width * 3 + 3) / 4 * 4);
    size_t pixelDataSize = rowStride * static_cast<size_t>(height);

    std::vector<uint8_t> raw(pixelDataSize);
    file.seekg(fileHeader.offset, std::ios::beg);
    file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));

    Image image = makeImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t index = static_cast<size_t>(y * rowStride + x * 3);
            uint8_t blue = raw[index];
            uint8_t green = raw[index + 1];
            uint8_t red = raw[index + 2];
            image.at(x, y) = RGB(red, green, blue);
        }
    }

    if (infoHeader.height > 0) {
        Image flipped = makeImage(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                flipped.at(x, y) = image.at(x, height - 1 - y);
            }
        }
        image = flipped;
    }

    return image;
}

void saveBMP(const std::string& path, const Image& image) {
    const int width = image.width;
    const int height = image.height;
    const int rowStride = ((width * 3 + 3) / 4) * 4;
    const int pixelDataSize = rowStride * height;

    BitmapFileHeader fileHeader{};
    fileHeader.type = 0x4D42;
    fileHeader.size = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + pixelDataSize;
    fileHeader.reserved1 = 0;
    fileHeader.reserved2 = 0;
    fileHeader.offset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

    BitmapInfoHeader infoHeader{};
    infoHeader.size = sizeof(BitmapInfoHeader);
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.planes = 1;
    infoHeader.bitCount = 24;
    infoHeader.compression = 0;
    infoHeader.imageSize = pixelDataSize;
    infoHeader.xPixelsPerMeter = 2835;
    infoHeader.yPixelsPerMeter = 2835;
    infoHeader.colorsUsed = 0;
    infoHeader.colorsImportant = 0;

    std::vector<uint8_t> rawData(static_cast<size_t>(pixelDataSize), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const RGB& pixel = image.at(x, y);
            const size_t base = static_cast<size_t>(y * rowStride + x * 3);
            rawData[base] = pixel.b;
            rawData[base + 1] = pixel.g;
            rawData[base + 2] = pixel.r;
        }
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo escribir la imagen de salida: " + path);
    }

    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
    file.write(reinterpret_cast<const char*>(rawData.data()), static_cast<std::streamsize>(rawData.size()));
}
