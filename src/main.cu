#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::ostringstream message;                                    \
            message << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                    << " - " << cudaGetErrorString(error);                \
            throw std::runtime_error(message.str());                       \
        }                                                                  \
    } while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

std::string readToken(std::ifstream& file) {
    std::string token;
    char character;

    while (file.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character))) {
            continue;
        }

        if (character == '#') {
            std::string ignored;
            std::getline(file, ignored);
            continue;
        }

        token += character;
        break;
    }

    while (file.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character))) {
            break;
        }

        if (character == '#') {
            std::string ignored;
            std::getline(file, ignored);
            break;
        }

        token += character;
    }

    return token;
}

Image readPPM(const fs::path& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Unable to open input image: " +
                                 filename.string());
    }

    const std::string magic = readToken(file);
    if (magic != "P6") {
        throw std::runtime_error("Expected binary PPM (P6): " +
                                 filename.string());
    }

    const int width = std::stoi(readToken(file));
    const int height = std::stoi(readToken(file));
    const int maxValue = std::stoi(readToken(file));

    if (width <= 0 || height <= 0 || maxValue != 255) {
        throw std::runtime_error("Invalid PPM header: " +
                                 filename.string());
    }

    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * height * 3);

    file.read(reinterpret_cast<char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size()));

    if (!file) {
        throw std::runtime_error("Incomplete image data: " +
                                 filename.string());
    }

    return image;
}

void writePGM(const fs::path& filename,
              const std::vector<unsigned char>& pixels,
              int width,
              int height) {
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Unable to create output image: " +
                                 filename.string());
    }

    file << "P5\n";
    file << width << " " << height << "\n";
    file << "255\n";

    file.write(reinterpret_cast<const char*>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));

    if (!file) {
        throw std::runtime_error("Unable to write output image: " +
                                 filename.string());
    }
}

__global__ void grayscaleKernel(const unsigned char* rgb,
                                unsigned char* grayscale,
                                int width,
                                int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pixel = y * width + x;
    const int rgbIndex = pixel * 3;

    const float red = static_cast<float>(rgb[rgbIndex]);
    const float green = static_cast<float>(rgb[rgbIndex + 1]);
    const float blue = static_cast<float>(rgb[rgbIndex + 2]);

    const float gray = 0.299f * red +
                       0.587f * green +
                       0.114f * blue;

    grayscale[pixel] =
        static_cast<unsigned char>(gray + 0.5f);
}

__constant__ float gaussianKernel[25];

__global__ void gaussianBlurKernel(const unsigned char* input,
                                    unsigned char* output,
                                    int width,
                                    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;

    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            const int sourceX = min(max(x + kx, 0), width - 1);
            const int sourceY = min(max(y + ky, 0), height - 1);

            const int sourceIndex = sourceY * width + sourceX;
            const int kernelIndex = (ky + 2) * 5 + (kx + 2);

            sum += static_cast<float>(input[sourceIndex]) *
                   gaussianKernel[kernelIndex];
        }
    }

    sum = fminf(fmaxf(sum, 0.0f), 255.0f);

    output[y * width + x] =
        static_cast<unsigned char>(sum + 0.5f);
}

std::vector<float> createGaussianKernel(float sigma) {
    std::vector<float> kernel(25);

    const float twoSigmaSquared = 2.0f * sigma * sigma;
    float total = 0.0f;

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            const float distanceSquared =
                static_cast<float>(x * x + y * y);

            const float value =
                std::exp(-distanceSquared / twoSigmaSquared);

            kernel[(y + 2) * 5 + (x + 2)] = value;
            total += value;
        }
    }

    for (float& value : kernel) {
        value /= total;
    }

    return kernel;
}

std::vector<fs::path> findImages(const fs::path& inputDirectory) {
    std::vector<fs::path> images;

    if (!fs::exists(inputDirectory)) {
        throw std::runtime_error(
            "Input directory does not exist: " +
            inputDirectory.string());
    }

    for (const auto& entry :
         fs::directory_iterator(inputDirectory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension =
            entry.path().extension().string();

        std::transform(extension.begin(),
                       extension.end(),
                       extension.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });

        if (extension == ".ppm") {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());

    return images;
}

struct Arguments {
    fs::path inputDirectory = "input";
    fs::path outputDirectory = "output";
    float sigma = 1.5f;
};

void printUsage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program
              << " --input <directory>"
              << " --output <directory>"
              << " --sigma <value>\n\n"
              << "Example:\n"
              << "  " << program
              << " --input input"
              << " --output output"
              << " --sigma 1.5\n";
}

Arguments parseArguments(int argc, char* argv[]) {
    Arguments arguments;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }

        if (option == "--input" && i + 1 < argc) {
            arguments.inputDirectory = argv[++i];
        } else if (option == "--output" && i + 1 < argc) {
            arguments.outputDirectory = argv[++i];
        } else if (option == "--sigma" && i + 1 < argc) {
            arguments.sigma = std::stof(argv[++i]);
        } else {
            throw std::runtime_error(
                "Unknown or incomplete option: " + option);
        }
    }

    if (arguments.sigma <= 0.0f) {
        throw std::runtime_error("Sigma must be greater than zero.");
    }

    return arguments;
}

int main(int argc, char* argv[]) {
    try {
        const Arguments arguments =
            parseArguments(argc, argv);

        fs::create_directories(arguments.outputDirectory);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0) {
            throw std::runtime_error(
                "No CUDA-capable GPU was detected.");
        }

        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

        std::cout << "============================================\n";
        std::cout << "CUDA GPU IMAGE PROCESSING PROJECT\n";
        std::cout << "============================================\n";
        std::cout << "GPU: " << properties.name << "\n";
        std::cout << "Compute Capability: "
                  << properties.major << "."
                  << properties.minor << "\n";
        std::cout << "Input Directory: "
                  << arguments.inputDirectory << "\n";
        std::cout << "Output Directory: "
                  << arguments.outputDirectory << "\n";
        std::cout << "Gaussian Sigma: "
                  << arguments.sigma << "\n\n";

        const std::vector<fs::path> images =
            findImages(arguments.inputDirectory);

        if (images.empty()) {
            throw std::runtime_error(
                "No .ppm images found in the input directory.");
        }

        std::cout << "Images found: "
                  << images.size() << "\n";

        const Image firstImage = readPPM(images.front());

        const int width = firstImage.width;
        const int height = firstImage.height;

        const size_t rgbBytes =
            static_cast<size_t>(width) * height * 3;

        const size_t grayBytes =
            static_cast<size_t>(width) * height;

        std::vector<unsigned char> hostRGB(rgbBytes);
        std::vector<unsigned char> hostGray(grayBytes);
        std::vector<unsigned char> hostBlur(grayBytes);

        unsigned char* deviceRGB = nullptr;
        unsigned char* deviceGray = nullptr;
        unsigned char* deviceBlur = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceRGB),
            rgbBytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceGray),
            grayBytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceBlur),
            grayBytes));

        const std::vector<float> kernel =
            createGaussianKernel(arguments.sigma);

        CUDA_CHECK(cudaMemcpyToSymbol(
            gaussianKernel,
            kernel.data(),
            kernel.size() * sizeof(float)));

        const dim3 blockSize(16, 16);

        const dim3 gridSize(
            (width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);

        cudaEvent_t start;
        cudaEvent_t stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        size_t processed = 0;

        for (const fs::path& imagePath : images) {
            const Image image = readPPM(imagePath);

            if (image.width != width ||
                image.height != height) {
                throw std::runtime_error(
                    "All input images must have identical dimensions.");
            }

            std::copy(image.pixels.begin(),
                      image.pixels.end(),
                      hostRGB.begin());

            CUDA_CHECK(cudaMemcpy(
                deviceRGB,
                hostRGB.data(),
                rgbBytes,
                cudaMemcpyHostToDevice));

            grayscaleKernel<<<gridSize, blockSize>>>(
                deviceRGB,
                deviceGray,
                width,
                height);

            CUDA_CHECK(cudaGetLastError());

            gaussianBlurKernel<<<gridSize, blockSize>>>(
                deviceGray,
                deviceBlur,
                width,
                height);

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(
                hostBlur.data(),
                deviceBlur,
                grayBytes,
                cudaMemcpyDeviceToHost));

            fs::path outputName =
                imagePath.stem().string() + "_processed.pgm";

            writePGM(
                arguments.outputDirectory / outputName,
                hostBlur,
                width,
                height);

            ++processed;

            if (processed % 10 == 0 ||
                processed == images.size()) {
                std::cout << "Processed "
                          << processed << "/"
                          << images.size()
                          << " images\n";
            }
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(deviceRGB));
        CUDA_CHECK(cudaFree(deviceGray));
        CUDA_CHECK(cudaFree(deviceBlur));

        std::cout << "\n============================================\n";
        std::cout << "PROCESSING COMPLETE\n";
        std::cout << "============================================\n";
        std::cout << "Images processed: "
                  << processed << "\n";
        std::cout << "Image dimensions: "
                  << width << " x "
                  << height << "\n";
        std::cout << "GPU pipeline time: "
                  << std::fixed
                  << std::setprecision(3)
                  << milliseconds << " ms\n";

        if (processed > 0) {
            std::cout << "Average GPU pipeline time/image: "
                      << milliseconds / processed
                      << " ms\n";
        }

        std::cout << "Output format: PGM\n";
        std::cout << "Status: SUCCESS\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: "
                  << error.what() << "\n";
        return 1;
    }
}
