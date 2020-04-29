#include <iostream>
#include <vector>
#include <string>

#include <libKitsunemimiPersistence/logger/logger.h>

#include <libKitsunemimiOpencl/opencl.h>

int main()
{
    Kitsunemimi::Persistence::initConsoleLogger(true);

    const size_t N = 1 << 20;

    // example kernel for task: c = a + b.
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

    // create config-object
    Kitsunemimi::Opencl::OpenClConfig config;
    config.kernelCode = kernelCode;
    config.kernelName = "add";

    // create data-object
    Kitsunemimi::Opencl::OpenClData data;
    data.range = N;

    // init buffer
    const uint32_t numberOfBlocks = ((N * sizeof(float)) / 4096) + 1;
    data.inputBuffer.push_back(Kitsunemimi::DataBuffer(numberOfBlocks));
    data.inputBuffer.push_back(Kitsunemimi::DataBuffer(numberOfBlocks));
    Kitsunemimi::allocateBlocks_DataBuffer(data.outputBuffer, (N / 4096) + 1);

    // set buffer-position
    data.inputBuffer[0].bufferPosition = N;
    data.inputBuffer[1].bufferPosition = N;
    data.outputBuffer.bufferPosition = N;

    // convert pointer
    float* a = static_cast<float*>(data.inputBuffer[0].data);
    float* b = static_cast<float*>(data.inputBuffer[1].data);

    // write intput dat into buffer
    for(uint32_t i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // run
    if(ocl.init(config)) {
        ocl.run(data);
    }

    // check result
    float* outputValues = static_cast<float*>(data.outputBuffer.data);
    // Should get '3' here.
    std::cout << outputValues[42] << std::endl;
}
