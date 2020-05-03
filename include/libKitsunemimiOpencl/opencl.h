#ifndef OPENCL_H
#define OPENCL_H

#include <iostream>
#include <vector>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
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
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct WorkerBuffer
{
    void* data = nullptr;
    uint64_t numberOfBytes;
    uint64_t numberOfObjects;
    bool isOutput = false;
    cl::Buffer clBuffer;

    WorkerBuffer() {}

    WorkerBuffer(const uint64_t numberOfObjects,
                 const uint64_t objectSize,
                 const bool isOutput = false)
    {
        this->numberOfBytes = numberOfObjects * objectSize;
        this->data = Kitsunemimi::alignedMalloc(4096, numberOfBytes);
        this->numberOfObjects = numberOfObjects;
        this->isOutput = isOutput;
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

class Opencl
{
public:
    Opencl();

    bool init(const OpenClConfig &config);

    void run(OpenClData &data);

private:
    std::vector<cl::Platform> m_platform;
    std::vector<cl::Device> m_device;
    cl::Context m_context;
    cl::Kernel m_kernel;
    cl::CommandQueue m_queue;

    void collectDevices(const OpenClConfig &config);
    bool build(const OpenClConfig &config);
};

}
}

#endif // OPENCL_H
