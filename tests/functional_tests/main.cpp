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
        "       global const float* a,\n"
        "       ulong n1,\n"
        "       global const float* b,\n"
        "       ulong n2,\n"
        "       global float* c,\n"
        "       ulong out\n"
        "       )\n"
        "{\n"
        "    size_t i = get_global_id(0);\n"
        "    if (i < n1) {\n"
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

    data.numberOfWg.x = N;
    data.threadsPerWg.x = 1;

    // init empty buffer
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float)));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float)));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float), true));

    // convert pointer
    float* a = static_cast<float*>(data.buffer[0].data);
    float* b = static_cast<float*>(data.buffer[1].data);

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
    float* outputValues = static_cast<float*>(data.buffer[2].data);
    // Should get '3' here.
    std::cout << outputValues[42] << std::endl;
}
