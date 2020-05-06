#ifndef OPENCL_H
#define OPENCL_H

#include <iostream>
#include <vector>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <libKitsunemimiOpencl/opencl_items.h>

namespace Kitsunemimi
{
namespace Opencl
{

class Opencl
{
public:
    Opencl();

    bool init(const OpenClConfig &config);
    bool copyToDevice(OpenClData &data);
    bool updateBuffer(WorkerBuffer &buffer);
    bool run(OpenClData &data);
    bool copyFromDevice(OpenClData &data);

    // getter for memory information
    uint64_t getLocalMemorySize();
    uint64_t getGlobalMemorySize();
    uint64_t getMaxMemAllocSize();

    // getter for work-group information
    uint64_t getMaxWorkGroupSize();
    WorkerDim getMaxWorkItemSize();
    uint64_t getMaxWorkItemDimension();

    // opencl objects
    // I left these public for the case, that there have to be some specific operations have to be
    // performed, which are not possible or available with the generic functions of this library.
    std::vector<cl::Platform> m_platform;
    std::vector<cl::Device> m_device;
    cl::Context m_context;
    cl::Kernel m_kernel;
    cl::CommandQueue m_queue;
    uint32_t m_argCounter = 0;

private:
    bool validateWorkerGroupSize(const OpenClData &data);
    void collectDevices(const OpenClConfig &config);
    bool build(const OpenClConfig &config);
};

}
}

#endif // OPENCL_H
