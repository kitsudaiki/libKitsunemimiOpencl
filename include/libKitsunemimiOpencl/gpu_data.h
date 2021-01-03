#ifndef GPU_DATA_H
#define GPU_DATA_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace Kitsunemimi
{
namespace Opencl
{
class GpuInterface;

struct WorkerDim
{
    uint64_t x = 1;
    uint64_t y = 1;
    uint64_t z = 1;
};

struct WorkerBuffer
{
    void* data = nullptr;
    uint64_t numberOfBytes = 0;
    uint64_t numberOfObjects = 0;
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

class GpuData
{
public:
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;

    GpuData();

    bool addBuffer(const std::string &name,
                   const WorkerBuffer &buffer);
    bool containsBuffer(const std::string &name);
    WorkerBuffer* getBuffer(const std::string &name);

private:
    friend GpuInterface;

    struct BufferLink
    {
        WorkerBuffer* buffer = nullptr;
        uint32_t bindedId = 0;
        uint8_t padding[4];
    };

    struct KernelDef
    {
        std::string id = "";
        std::string kernelCode = "";
        cl::Kernel kernel;
        std::map<std::string, BufferLink> bufferLinks;
        uint32_t localBufferSize = 0;
        uint32_t argumentCounter = 0;
    };

    std::map<std::string, WorkerBuffer> m_buffer;
    std::map<std::string, KernelDef> m_kernel;

    bool containsKernel(const std::string &name);
    KernelDef* getKernel(const std::string &name);
};

}
}

#endif // GPU_DATA_H
