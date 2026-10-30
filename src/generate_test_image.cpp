#include <iostream>
#include <stdexcept>

#include "bmp.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: generate_test_image.exe <output.bmp>\n";
        return 1;
    }

    try {
        Image image = generateTestImage(512, 512);
        saveBMP(argv[1], image);
        std::cout << "Imagen generada: " << argv[1] << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
