#ifndef OPENCL_H
#define OPENCL_H

#include <iostream>
#include <vector>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace Kitsunemimi
{
namespace Opencl
{

class Opencl
{
public:
    Opencl();

    bool init();
    bool run(const std::string &kernelCode);

private:
    std::vector<cl::Platform> m_platform;
    std::vector<cl::Device> m_device;
    cl::Context m_context;
};

}
}

#endif // OPENCL_H
