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
    std::vector<cl::Platform> platform;
    cl::Context context;
    std::vector<cl::Device> device;
};

}
}

#endif // OPENCL_H
