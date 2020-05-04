#include <libKitsunemimiOpencl/opencl.h>

#include <libKitsunemimiPersistence/logger/logger.h>

namespace Kitsunemimi
{
namespace Opencl
{

/**
 * @brief constructor
 */
Opencl::Opencl() {}

/**
 * @brief initialize opencl
 *
 * @param config object with config-parameter
 *
 * @return true, if creation was successful, else false
 */
bool
Opencl::init(const OpenClConfig &config)
{
    try
    {
        // get all available opencl platforms
        cl::Platform::get(&m_platform);
        if(m_platform.empty())
        {
            LOG_ERROR("OpenCL platforms not found.");
            return false;
        }

        LOG_INFO("number of OpenCL platforms: " + std::to_string(m_platform.size()));

        // get devices from all available platforms
        collectDevices(config);

        // check if there were devices found
        if(m_device.empty())
        {
            LOG_ERROR("No OpenCL device found.");
            return false;
        }

        LOG_INFO("choosen OpenCL device: " + m_device[0].getInfo<CL_DEVICE_NAME>());

        // create command queue.
        m_queue = cl::CommandQueue(m_context, m_device[0]);

        // build kernel
        const bool buildResult = build(config);

        return buildResult;
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
 * @brief Opencl::copyToDevice
 * @param data
 * @return
 */
bool
Opencl::copyToDevice(OpenClData &data)
{
    uint32_t argCounter = 0;

    // send input to device
    for(uint64_t i = 0; i < data.buffer.size(); i++)
    {
        WorkerBuffer buffer = data.buffer.at(i);

        if(buffer.isOutput)
        {
            data.buffer[i].clBuffer = cl::Buffer(m_context,
                                                 CL_MEM_READ_WRITE,
                                                 buffer.numberOfBytes);
            m_kernel.setArg(argCounter, data.buffer[i].clBuffer);
            argCounter++;
        }
        else
        {
            data.buffer[i].clBuffer = cl::Buffer(m_context,
                                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 buffer.numberOfBytes,
                                                 buffer.data);
            m_kernel.setArg(argCounter, data.buffer[i].clBuffer);
            argCounter++;
        }

        m_kernel.setArg(argCounter, static_cast<cl_ulong>(buffer.numberOfObjects));
        argCounter++;
    }

    return true;
}

/**
 * @brief run kernel with input
 *
 * @param data input-data for the run
 */
bool
Opencl::run(OpenClData &data)
{
    // convert ranges
    const cl::NDRange globalRange = cl::NDRange(data.numberOfWg.x * data.threadsPerWg.x,
                                                data.numberOfWg.y * data.threadsPerWg.y,
                                                data.numberOfWg.z * data.threadsPerWg.z);
    const cl::NDRange localRange = cl::NDRange(data.threadsPerWg.x,
                                               data.threadsPerWg.y,
                                               data.threadsPerWg.z);

    // launch kernel on the device
    m_queue.enqueueNDRangeKernel(m_kernel,
                                 cl::NullRange,
                                 globalRange,
                                 localRange);

    return true;
}

/**
 * @brief Opencl::copyFromDevice
 * @param data
 */
bool Opencl::copyFromDevice(OpenClData &data)
{
    // get output back from device
    for(uint64_t i = 0; i < data.buffer.size(); i++)
    {
        if(data.buffer.at(i).isOutput)
        {
            // copy result back to host
            m_queue.enqueueReadBuffer(data.buffer[i].clBuffer,
                                      CL_TRUE,
                                      0,
                                      data.buffer[i].numberOfBytes,
                                      data.buffer[i].data);
        }
    }

    return true;
}

/**
 * @brief Opencl::getVendor
 * @return
 */
const std::string
Opencl::getVendor()
{
    if(m_platform.size() == 0) {
        return "";
    }

    std::string vendor;
    m_platform.at(0).getInfo(CL_PLATFORM_VENDOR, &vendor);

    return vendor;
}

/**
 * @brief Opencl::getSizeOfLocalMemory
 * @return
 */
uint64_t
Opencl::getLocalMemorySize()
{
    if(m_device.size() == 0) {
        return 0;
    }

    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief Opencl::getGlobalMemorySize
 * @return
 */
uint64_t
Opencl::getGlobalMemorySize_total()
{
    if(m_device.size() == 0) {
        return 0;
    }

    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief Opencl::getGlobalMemorySize_available
 * @return
 */
uint64_t
Opencl::getGlobalMemorySize_free()
{
    if(m_device.size() == 0
            || getVendor() != "AMD")
    {
        return 0;
    }

    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, &size);

    return size;
}

/**
 * @brief Opencl::getMaxMemAllocSize
 * @return
 */
uint64_t
Opencl::getMaxMemAllocSize()
{
    if(m_device.size() == 0) {
        return 0;
    }

    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);

    return size;
}

/**
 * @brief Opencl::getMaxWorkgroupSize
 * @return
 */
uint64_t
Opencl::getMaxWorkGroupSize()
{
    if(m_device.size() == 0) {
        return 0;
    }

    size_t size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);

    return size;
}

/**
 * @brief Opencl::getMaxThreadNumber
 * @return
 */
WorkerDim
Opencl::getMaxWorkItemSize()
{
    if(m_device.size() == 0) {
        return WorkerDim();
    }

    size_t size[3];
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size);

    uint64_t dimension = getMaxWorkItemDimension();
    WorkerDim result;
    if(dimension > 0) {
        result.x = size[0];
    }
    if(dimension > 1) {
        result.y = size[1];
    }
    if(dimension > 2) {
        result.z = size[2];
    }

    return result;
}

/**
 * @brief Opencl::getMaxTheadDimension
 * @return
 */
uint64_t
Opencl::getMaxWorkItemDimension()
{
    if(m_device.size() == 0) {
        return 0;
    }

    cl_uint size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &size);

    return size;
}

/**
 * @brief collect all available devices
 *
 * @param config object with config-parameter
 */
void
Opencl::collectDevices(const OpenClConfig &config)
{
    // get available platforms
    std::vector<cl::Platform>::const_iterator plat_it;
    for(plat_it = m_platform.begin();
        plat_it != m_platform.end();
        plat_it++)
    {
        // get available devices of the selected platform
        std::vector<cl::Device> pldev;
        plat_it->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
        LOG_INFO("number of OpenCL devices: " + std::to_string(pldev.size()));

        // select devices within the platform
        std::vector<cl::Device>::const_iterator dev_it;
        for(dev_it = pldev.begin();
            dev_it != pldev.end();
            dev_it++)
        {
            // check if device is available
            if(dev_it->getInfo<CL_DEVICE_AVAILABLE>())
            {
                if(config.requiresDoublePrecision)
                {
                    // check for double precision support
                    const std::string ext = dev_it->getInfo<CL_DEVICE_EXTENSIONS>();
                    if(ext.find("cl_khr_fp64") != std::string::npos
                        && ext.find("cl_amd_fp64") != std::string::npos)
                    {
                        m_device.push_back(*dev_it);
                        m_context = cl::Context(m_device);
                    }
                }
                else
                {
                    // add all devices
                    m_device.push_back(*dev_it);
                    m_context = cl::Context(m_device);
                }
            }

            // if a maximum number of devices was selected, then break the loop
            if(m_device.size() == config.maxNumberOfDevice
                    && config.maxNumberOfDevice > 0)
            {
                return;
            }
        }
    }
}

/**
 * @brief build kernel-code
 *
 * @param config object with config-parameter
 *
 * @return true, if successful, else false
 */
bool
Opencl::build(const OpenClConfig &config)
{
    // compile opencl program for found device.
    const std::pair<const char*, size_t> kernelCode = std::make_pair(config.kernelCode.c_str(),
                                                                     config.kernelCode.size());
    const cl::Program::Sources source = cl::Program::Sources(1, kernelCode);
    cl::Program program(m_context, source);

    try
    {
        // build for all selected devices
        program.build(m_device);
    }
    catch(const cl::Error&)
    {
        LOG_ERROR("OpenCL compilation error\n    "
                  + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device[0]));
        return false;
    }

    // create kernel
    m_kernel = cl::Kernel(program, config.kernelName.c_str());

    return true;
}

}
}
