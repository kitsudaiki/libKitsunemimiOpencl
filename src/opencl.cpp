#include <libKitsunemimiOpencl/opencl.h>

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
        cl::Platform::get(&platform);
        if(platform.empty())
        {
            std::cerr << "OpenCL platforms not found." << std::endl;
            return 1;
        }
        std::cout<<"number of platforms: "<<platform.size()<<std::endl;

        // Get first available GPU device which supports double precision.
        for(auto p = platform.begin(); device.empty() && p != platform.end(); p++)
        {
            std::vector<cl::Device> pldev;

            p->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
            std::cout<<"number of devices: "<<pldev.size()<<std::endl;

            for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++)
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

                device.push_back(*d);
                context = cl::Context(device);
            }
        }

        if(device.empty())
        {
            //std::cerr << "GPUs with double precision not found." << std::endl;
            std::cerr << "No device found." << std::endl;
            return 1;
        }

        std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    }
    catch(const cl::Error &err)
    {
        std::cerr
            << "OpenCL error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
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
        cl::CommandQueue queue(context, device[0]);

        // Compile OpenCL program for found device.
        const cl::Program::Sources source = cl::Program::Sources(1,
                                                                 std::make_pair(kernelCode.c_str(),
                                                                                kernelCode.size()));
        cl::Program program(context, source);

        try
        {
            program.build(device);
        }
        catch(const cl::Error&)
        {
            std::cerr
                << "OpenCL compilation error" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
                << std::endl;
            return 1;
        }

        cl::Kernel add(program, "add");

        // Prepare input data.
        std::vector<float> a(N, 1);
        std::vector<float> b(N, 2);
        std::vector<float> c(N);

        // Allocate device buffers and transfer input data to device.
        cl::Buffer A(context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     a.size() * sizeof(float),
                     a.data());

        cl::Buffer B(context,
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     b.size() * sizeof(float),
                     b.data());

        cl::Buffer C(context,
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
        std::cerr
            << "OpenCL error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
        return false;
    }

    return true;
}

}
}
