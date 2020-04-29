#include <libKitsunemimiOpencl/opencl.h>

#include <libKitsunemimiPersistence/logger/logger.h>

namespace Kitsunemimi
{
namespace Opencl
{

Opencl::Opencl() {}

/**
 * @brief opencl::init
 * @return
 */
bool
Opencl::init()
{
    try
    {
        // Get list of OpenCL platforms.
        cl::Platform::get(&m_platform);
        if(m_platform.empty())
        {
            LOG_ERROR("OpenCL platforms not found.");
            return false;
        }

        LOG_INFO("number of OpenCL platforms: " + std::to_string(m_platform.size()));

        // Get first available GPU device which supports double precision.
        for(auto p = m_platform.begin(); m_device.empty() && p != m_platform.end(); p++)
        {
            std::vector<cl::Device> pldev;

            p->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
            LOG_INFO("number of OpenCL devices: " + std::to_string(pldev.size()));

            for(auto d = pldev.begin(); m_device.empty() && d != pldev.end(); d++)
            {
                if(!d->getInfo<CL_DEVICE_AVAILABLE>()) {
                    continue;
                }

                std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

                // check for double precision support
                //if(ext.find("cl_khr_fp64") == std::string::npos
                //    && ext.find("cl_amd_fp64") == std::string::npos)
                //{
                //     continue;
                //}

                m_device.push_back(*d);
                m_context = cl::Context(m_device);
            }
        }

        if(m_device.empty())
        {
            LOG_ERROR("No OpenCL device found.");
            return false;
        }

        LOG_INFO("choosen OpenCL device: " + m_device[0].getInfo<CL_DEVICE_NAME>());
    }
    catch(const cl::Error &err)
    {
        LOG_ERROR("OpenCL error: "
                  + std::string(err.what())
                  + "("
                  + std::to_string(err.err())
                  + ")");
        return false;
    }

    return true;
}

/**
 * @brief opencl::run
 * @param kernelCode
 * @return
 */
bool
Opencl::run(const std::string &kernelCode)
{
    const size_t N = 1 << 20;

    try
    {
        // Create command queue.
        cl::CommandQueue queue(m_context, m_device[0]);

        // Compile OpenCL program for found device.
        const cl::Program::Sources source = cl::Program::Sources(1,
                                                                 std::make_pair(kernelCode.c_str(),
                                                                                kernelCode.size()));
        cl::Program program(m_context, source);

        try
        {
            program.build(m_device);
        }
        catch(const cl::Error&)
        {
            LOG_ERROR("OpenCL compilation error\n    "
                      + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device[0]));
            return false;
        }

        cl::Kernel add(program, "add");

        // Prepare input data.
        std::vector<float> a(N, 1);
        std::vector<float> b(N, 2);
        std::vector<float> c(N);

        // Allocate device buffers and transfer input data to device.
        cl::Buffer A(m_context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     a.size() * sizeof(float),
                     a.data());

        cl::Buffer B(m_context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     b.size() * sizeof(float),
                     b.data());

        cl::Buffer C(m_context,
                     CL_MEM_READ_WRITE,
                     c.size() * sizeof(float));

        // Set kernel parameters.
        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        // Launch kernel on the compute device.
        queue.enqueueNDRangeKernel(add,
                                   cl::NullRange,
                                   N,
                                   cl::NullRange);

        // Get result back to host.
        queue.enqueueReadBuffer(C,
                                CL_TRUE,
                                0,
                                c.size() * sizeof(float),
                                c.data());

        // Should get '3' here.
        std::cout << c[42] << std::endl;
    }
    catch(const cl::Error &err)
    {
        LOG_ERROR("OpenCL error: "
                  + std::string(err.what())
                  + "("
                  + std::to_string(err.err())
                  + ")");
        return false;
    }

    return true;
}

}
}
