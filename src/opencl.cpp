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
Opencl::init(const OpenClConfig &config)
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

        collectDevices(config);

        if(m_device.empty())
        {
            LOG_ERROR("No OpenCL device found.");
            return false;
        }

        LOG_INFO("choosen OpenCL device: " + m_device[0].getInfo<CL_DEVICE_NAME>());

        return build(config);
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
}

/**
 * @brief Opencl::run
 * @param data
 * @return
 */
bool
Opencl::run(OpenClData &data)
{
    uint32_t argCounter = 0;

    m_kernel.setArg(argCounter, static_cast<cl_ulong>(data.range));
    argCounter++;

    for(uint64_t i = 0; i < data.inputBuffer.size(); i++)
    {
        cl::Buffer input(m_context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         data.inputBuffer.at(i).bufferPosition,
                         data.inputBuffer.at(i).data);
        m_kernel.setArg(argCounter, input);
        argCounter++;
    }

    cl::Buffer output(m_context,
                      CL_MEM_READ_WRITE,
                      data.outputBuffer.bufferPosition);
    m_kernel.setArg(argCounter, output);
    argCounter++;

    // Set kernel parameters.


    // Launch kernel on the compute device.
    m_queue.enqueueNDRangeKernel(m_kernel,
                                 cl::NullRange,
                                 data.range,
                                 cl::NullRange);

    // Get result back to host.
    m_queue.enqueueReadBuffer(output,
                              CL_TRUE,
                              0,
                              data.outputBuffer.bufferPosition,
                              data.outputBuffer.data);

    return true;
}

/**
 * @brief Opencl::collectDevices
 * @param config
 */
void
Opencl::collectDevices(const OpenClConfig &config)
{
    // get available devices
    std::vector<cl::Platform>::const_iterator plat_it;
    for(plat_it = m_platform.begin();
        plat_it != m_platform.end();
        plat_it++)
    {
        std::vector<cl::Device> pldev;

        plat_it->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
        LOG_INFO("number of OpenCL devices: " + std::to_string(pldev.size()));

        std::vector<cl::Device>::const_iterator dev_it;
        for(dev_it = pldev.begin();
            dev_it != pldev.end();
            dev_it++)
        {
            if(dev_it->getInfo<CL_DEVICE_AVAILABLE>())
            {
                // check for double precision support
                if(config.requiresDoublePrecision)
                {
                    std::string ext = dev_it->getInfo<CL_DEVICE_EXTENSIONS>();
                    if(ext.find("cl_khr_fp64") != std::string::npos
                        && ext.find("cl_amd_fp64") != std::string::npos)
                    {
                        m_device.push_back(*dev_it);
                        m_context = cl::Context(m_device);
                    }
                }
                else
                {
                    m_device.push_back(*dev_it);
                    m_context = cl::Context(m_device);
                }
            }

            if(m_device.size() == config.maxNumberOfDevice
                    && config.maxNumberOfDevice > 0)
            {
                return;
            }
        }
    }
}

/**
 * @brief opencl::build
 * @param kernelCode
 * @return
 */
bool
Opencl::build(const OpenClConfig &config)
{
    // Create command queue.
    assert(m_device.size() > 0);
    m_queue = cl::CommandQueue(m_context, m_device[0]);

    // Compile OpenCL program for found device.
    const std::pair<const char*, size_t> kernelCode = std::make_pair(config.kernelCode.c_str(),
                                                                     config.kernelCode.size());
    const cl::Program::Sources source = cl::Program::Sources(1, kernelCode);
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

    m_kernel = cl::Kernel(program, config.kernelName.c_str());

    return true;
}

}
}
