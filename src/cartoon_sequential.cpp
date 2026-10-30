#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bmp.h"
#include "cartoon_common.h"

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Uso: cartoon_sequential.exe <input.bmp> <output.bmp> <filterSize> <levels> <threshold>\n";
        return 1;
    }

    try {
        const std::string inputPath = argv[1];
        const std::string outputPath = argv[2];
        const int kernelSize = std::stoi(argv[3]);
        const int levels = std::stoi(argv[4]);
        const int threshold = std::stoi(argv[5]);

        auto start = std::chrono::steady_clock::now();
        Image input = loadBMP(inputPath);
        Image output = cartoonizeSequential(input, kernelSize, levels, threshold);
        saveBMP(outputPath, output);
        auto end = std::chrono::steady_clock::now();

        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Version secuencial: " << millis << " ms\n";
        std::cout << "Salida generada en: " << outputPath << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
