#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include "bmp.h"
#include "cartoon_common.h"

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Uso: cartoon_mpi.exe <input.bmp> <output.bmp> <filterSize> <levels> <threshold>\n";
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    try {
        const std::string inputPath = argv[1];
        const std::string outputPath = argv[2];
        const int kernelSize = std::stoi(argv[3]);
        const int levels = std::stoi(argv[4]);
        const int threshold = std::stoi(argv[5]);

        Image input;
        int width = 0;
        int height = 0;

        if (rank == 0) {
            input = loadBMP(inputPath);
            width = input.width;
            height = input.height;
        }

        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> counts(worldSize, 0);
        std::vector<int> displacements(worldSize, 0);
        if (rank == 0) {
            int baseRows = height / worldSize;
            int remainder = height % worldSize;
            int start = 0;
            for (int r = 0; r < worldSize; ++r) {
                counts[r] = baseRows + (r < remainder ? 1 : 0);
                displacements[r] = start;
                start += counts[r];
            }
        }

        std::vector<int> rowsState(worldSize, 0);
        MPI_Bcast(counts.data(), static_cast<int>(counts.size()), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displacements.data(), static_cast<int>(displacements.size()), MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<RGB> localPixels;
        const int rowsForThisRank = counts[rank];
        localPixels.resize(static_cast<size_t>(rowsForThisRank * std::max(1, width)));

        std::vector<uint8_t> sendBuffer;
        if (rank == 0) {
            sendBuffer.resize(static_cast<size_t>(height * width * 3));
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const RGB& p = input.at(x, y);
                    size_t idx = static_cast<size_t>((y * width + x) * 3);
                    sendBuffer[idx] = p.r;
                    sendBuffer[idx + 1] = p.g;
                    sendBuffer[idx + 2] = p.b;
                }
            }
        }

        std::vector<int> receivedCounts(worldSize, 0);
        for (int r = 0; r < worldSize; ++r) {
            receivedCounts[r] = counts[r] * width * 3;
        }

        std::vector<int> receivedDisplacements(worldSize, 0);
        for (int r = 1; r < worldSize; ++r) {
            receivedDisplacements[r] = receivedDisplacements[r - 1] + receivedCounts[r - 1];
        }

        std::vector<uint8_t> receivedBuffer(static_cast<size_t>(height * width * 3));
        if (rank == 0) {
            MPI_Scatterv(sendBuffer.data(), receivedCounts.data(), receivedDisplacements.data(), MPI_UNSIGNED_CHAR,
                         MPI_IN_PLACE, 0, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        } else {
            std::vector<uint8_t> myChunk(static_cast<size_t>(rowsForThisRank * width * 3));
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                         myChunk.data(), static_cast<int>(myChunk.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            for (int row = 0; row < rowsForThisRank; ++row) {
                for (int x = 0; x < width; ++x) {
                    size_t idx = static_cast<size_t>((row * width + x) * 3);
                    localPixels[static_cast<size_t>(row * width + x)] = RGB(myChunk[idx], myChunk[idx + 1], myChunk[idx + 2]);
                }
            }
        }

        if (rank == 0) {
            for (int row = 0; row < rowsForThisRank; ++row) {
                for (int x = 0; x < width; ++x) {
                    size_t idx = static_cast<size_t>((row * width + x) * 3);
                    localPixels[static_cast<size_t>(row * width + x)] = RGB(sendBuffer[idx], sendBuffer[idx + 1], sendBuffer[idx + 2]);
                }
            }
        }

        Image localImage = makeImage(width, rowsForThisRank);
        localImage.pixels = localPixels;

        auto start = std::chrono::steady_clock::now();
        Image localOutput = cartoonizeMPI(localImage, kernelSize, levels, threshold, rank, worldSize);
        auto end = std::chrono::steady_clock::now();

        std::vector<uint8_t> flattened(static_cast<size_t>(localOutput.width * localOutput.height * 3));
        for (int y = 0; y < localOutput.height; ++y) {
            for (int x = 0; x < localOutput.width; ++x) {
                const RGB& p = localOutput.at(x, y);
                size_t idx = static_cast<size_t>((y * localOutput.width + x) * 3);
                flattened[idx] = p.r;
                flattened[idx + 1] = p.g;
                flattened[idx + 2] = p.b;
            }
        }

        std::vector<int> recvCounts(worldSize, 0);
        for (int r = 0; r < worldSize; ++r) {
            recvCounts[r] = counts[r] * width * 3;
        }
        std::vector<int> recvDisplacements(worldSize, 0);
        for (int r = 1; r < worldSize; ++r) {
            recvDisplacements[r] = recvDisplacements[r - 1] + recvCounts[r - 1];
        }

        std::vector<uint8_t> gathered(static_cast<size_t>(height * width * 3), 0);
        MPI_Gatherv(flattened.data(), static_cast<int>(flattened.size()), MPI_UNSIGNED_CHAR,
                    gathered.data(), recvCounts.data(), recvDisplacements.data(), MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            Image finalImage = makeImage(width, height);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    size_t idx = static_cast<size_t>((y * width + x) * 3);
                    finalImage.at(x, y) = RGB(gathered[idx], gathered[idx + 1], gathered[idx + 2]);
                }
            }
            saveBMP(outputPath, finalImage);
            std::cout << "Version MPI: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
            std::cout << "Salida generada en: " << outputPath << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
