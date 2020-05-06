#ifndef OPENCL_ITEMS_H
#define OPENCL_ITEMS_H

#include <iostream>
#include <vector>
#include <string>

#include <CL/cl.hpp>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

namespace Kitsunemimi
{
namespace Opencl
{

enum DeviceType
{
    GPU_TYPE,
    CPU_TYPE,
    ALL_TYPE
};

struct WorkerDim
{
    uint64_t x = 1;
    uint64_t y = 1;
    uint64_t z = 1;
};

struct WorkerBuffer
{
    void* data = nullptr;
    uint64_t numberOfBytes;
    uint64_t numberOfObjects;
    bool isOutput = false;
    bool useHostPtr = false;
    cl::Buffer clBuffer;

    WorkerBuffer() {}

    WorkerBuffer(const uint64_t numberOfObjects,
                 const uint64_t objectSize,
                 const bool isOutput = false,
                 const bool useHostPtr = false)
    {
        this->numberOfBytes = numberOfObjects * objectSize;
        this->data = Kitsunemimi::alignedMalloc(4096, numberOfBytes);
        this->numberOfObjects = numberOfObjects;
        this->isOutput = isOutput;
        this->useHostPtr = useHostPtr;
    }
};

struct OpenClConfig
{
    std::string kernelCode = "";
    std::string kernelName = "";

    DeviceType type = GPU_TYPE;
    uint32_t maxNumberOfDevice = 1; // 0 = all
    bool requiresDoublePrecision = false;
};

struct OpenClData
{
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;

    std::vector<WorkerBuffer> buffer;
};

}
}

#endif // OPENCL_ITEMS_H
