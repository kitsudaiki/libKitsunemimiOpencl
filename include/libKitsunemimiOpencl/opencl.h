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
