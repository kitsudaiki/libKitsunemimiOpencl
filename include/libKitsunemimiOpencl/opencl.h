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
    uint64_t range = 0;
    std::vector<DataBuffer> inputBuffer;
    DataBuffer outputBuffer;
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
