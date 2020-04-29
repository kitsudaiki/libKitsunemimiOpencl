#include <iostream>
#include <vector>
#include <string>

#include <libKitsunemimiPersistence/logger/logger.h>

#include <libKitsunemimiOpencl/opencl.h>

int main()
{
    Kitsunemimi::Persistence::initConsoleLogger(true);

    const size_t N = 1 << 20;

    // Compute c = a + b.
    const std::string kernelCode =
        "kernel void add(\n"
        "       ulong n,\n"
        "       global const float *a,\n"
        "       global const float *b,\n"
        "       global float *c\n"
        "       )\n"
        "{\n"
        "    size_t i = get_global_id(0);\n"
        "    if (i < n) {\n"
        "       c[i] = a[i] + b[i];\n"
        "    }\n"
        "}\n";

    Kitsunemimi::Opencl::Opencl ocl;

    Kitsunemimi::Opencl::OpenClConfig config;
    config.kernelCode = kernelCode;
    config.kernelName = "add";

    Kitsunemimi::Opencl::OpenClData data;
    data.range = N;

    data.inputBuffer.push_back(Kitsunemimi::DataBuffer((N / 4096) + 1));
    data.inputBuffer.push_back(Kitsunemimi::DataBuffer((N / 4096) + 1));
    Kitsunemimi::allocateBlocks_DataBuffer(data.outputBuffer, (N / 4096) + 1);
    data.outputBuffer.bufferPosition = N;

    const float a = 1;
    const float b = 2;

    for(uint32_t i = 0; i < N; i++)
    {
        Kitsunemimi::addObject_DataBuffer(data.inputBuffer[0], &a);
        Kitsunemimi::addObject_DataBuffer(data.inputBuffer[1], &b);
    }

    assert(data.inputBuffer[0].bufferPosition == N * sizeof(float));
    assert(data.inputBuffer[1].bufferPosition == N * sizeof(float));

    if(ocl.init(config)) {
        ocl.run(data);
    }

    float* outputValues = static_cast<float*>(data.outputBuffer.data);
    // Should get '3' here.
    std::cout << outputValues[42] << std::endl;
}
