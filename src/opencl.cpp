/**
 * @file        opencl.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2020 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

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
 * @brief destructor to close at least the device-connection
 */
Opencl::~Opencl()
{
    OpenClData emptyData;
    closeDevice(emptyData);
}

/**
 * @brief initialize opencl
 *
 * @param config object with config-parameter
 *
 * @return true, if creation was successful, else false
 */
bool
Opencl::initDevice()
{
    LOG_DEBUG("initialize OpenCL device");

    // precheck
    if(m_device.size() > 0)
    {
        LOG_ERROR("device already initialized.");
        return false;
    }

    try
    {
        // get all available opencl platforms
        cl::Platform::get(&m_platform);
        if(m_platform.empty())
        {
            LOG_ERROR("No OpenCL platforms found.");
            return false;
        }

        LOG_DEBUG("number of OpenCL platforms: " + std::to_string(m_platform.size()));

        // get devices from all available platforms
        collectDevices();

        // check if there were devices found
        if(m_device.empty())
        {
            LOG_ERROR("No OpenCL device found.");
            return false;
        }

        LOG_DEBUG("choosen OpenCL device: " + m_device[0].getInfo<CL_DEVICE_NAME>());

        // create command queue.
        m_queue = cl::CommandQueue(m_context, m_device[0]);

        return true;
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
 * @brief copy data from host to device
 *
 * @param data object with all data
 *
 * @return true, if successful, else false
 */
bool
Opencl::initCopyToDevice(OpenClData &data)
{
    LOG_DEBUG("initial data transfer to OpenCL device");

    // precheck
    if(m_device.size() == 0)
    {
        LOG_ERROR("no device initialized.");
        return false;
    }

    // send input to device
    for(uint64_t i = 0; i < data.buffer.size(); i++)
    {
        WorkerBuffer buffer = data.buffer.at(i);

        LOG_DEBUG("copy data to device: "
                  + std::to_string(buffer.numberOfBytes)
                  + " Bytes");

        // check buffer
        if(buffer.numberOfBytes == 0
                || buffer.numberOfObjects == 0
                || buffer.data == nullptr)
        {
            LOG_ERROR("failed to copy data to device, because buffer number "
                      + std::to_string(i)
                      + " has size 0 or is not initialized.");
            return false;
        }

        // create flag for memory handling
        cl_mem_flags flags = 0;
        if(buffer.isOutput)
        {
            flags = CL_MEM_READ_WRITE;
            if(buffer.useHostPtr) {
                flags = flags | CL_MEM_USE_HOST_PTR;
            } else {
                flags = flags | CL_MEM_COPY_HOST_PTR;
            }
        }
        else
        {
            if(buffer.useHostPtr) {
                flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
            } else {
                flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            }
        }

        // send data or reference to device
        data.buffer[i].clBuffer = cl::Buffer(m_context,
                                             flags,
                                             buffer.numberOfBytes,
                                             buffer.data);
    }

    return true;
}

/**
 * @brief Opencl::addKernel
 * @param id
 * @param kernelCode
 * @return
 */
bool
Opencl::addKernel(const std::string &id,
                  const std::string &kernelCode)
{
    LOG_DEBUG("add kernel with id: " + id);

    // TODO: check if already registerd
    KernelDef def;
    def.id = id;
    def.kernelCode = kernelCode;

    try
    {
        const bool buildResult = build(def);
        if(buildResult == false) {
            return false;
        }

        m_kernel.insert(std::make_pair(id, def));
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
 * @brief Opencl::bindKernelToBuffer
 * @param kernelId
 * @param bufferId
 * @return
 */
bool
Opencl::bindKernelToBuffer(const std::string &kernelId,
                           const uint32_t bufferId,
                           OpenClData &data)
{
    LOG_DEBUG("bind buffer number " + std::to_string(bufferId) + " kernel with id: " + kernelId);

    std::map<std::string, KernelDef>::iterator it;
    it = m_kernel.find(kernelId);

    if(it == m_kernel.end())
    {
        LOG_ERROR("no kernel with name " + kernelId + " found");
        return false;
    }

    if(data.buffer.size() <= bufferId)
    {
        LOG_ERROR("no buffer with id " + std::to_string(bufferId) + " found");
        return false;
    }

    WorkerBuffer* buffer = &data.buffer[bufferId];

    const uint32_t argNumber = static_cast<uint32_t>(it->second.bufferLinks.size() * 2);
    LOG_DEBUG("bind buffer number "
              + std::to_string(bufferId)
              + " to argument nubmer "
              + std::to_string(argNumber));
    it->second.kernel.setArg(argNumber, buffer->clBuffer);
    LOG_DEBUG("bind size value of buffer number "
              + std::to_string(bufferId)
              + " to argument nubmer "
              + std::to_string(argNumber + 1));
    it->second.kernel.setArg(argNumber + 1, static_cast<cl_ulong>(buffer->numberOfObjects));

    it->second.bufferLinks.push_back(buffer);

    return true;
}

/**
 * @brief Opencl::setLocalMemory
 * @param kernelId
 * @param localMemorySize
 * @return
 */
bool
Opencl::setLocalMemory(const std::string &kernelId,
                       const uint32_t localMemorySize)
{
    std::map<std::string, KernelDef>::iterator it;
    it = m_kernel.find(kernelId);

    if(it == m_kernel.end())
    {
        LOG_ERROR("no kernel with name " + kernelId + " found");
        return false;
    }

    const uint32_t argNumber = static_cast<uint32_t>(it->second.bufferLinks.size()) * 2;

    it->second.kernel.setArg(argNumber, localMemorySize, nullptr);
    it->second.kernel.setArg(argNumber + 1, static_cast<cl_ulong>(localMemorySize));

    return true;
}


/**
 * @brief update data inside the buffer on the device
 *
 * @param buffer worker-buffer-object with the actual data
 * @param numberOfObjects number of objects to copy
 * @param offset offset in buffer on device
 *
 * @return false, if copy failed of buffer is output-buffer, else true
 */
bool
Opencl::updateBufferOnDevice(const std::string &kernelId,
                             const uint32_t bufferId,
                             uint64_t numberOfObjects,
                             const uint64_t offset)
{
    LOG_DEBUG("update buffer on OpenCL device");

    std::map<std::string, KernelDef>::iterator it;
    it = m_kernel.find(kernelId);

    if(it == m_kernel.end())
    {
        LOG_ERROR("no kernel with name " + kernelId + " found");
        return false;
    }

    if(it->second.bufferLinks.size() <= bufferId)
    {
        LOG_ERROR("no buffer with id " + std::to_string(bufferId) + " found");
        return false;
    }

    WorkerBuffer* buffer = it->second.bufferLinks[bufferId];

    const uint64_t objectSize = buffer->numberOfBytes / buffer->numberOfObjects;

    // precheck
    if(m_device.size() == 0)
    {
        LOG_ERROR("no device initialized.");
        return false;
    }

    // check if buffer is output-buffer
    if(buffer->isOutput) {
        return false;
    }

    // set size with value of the buffer, if size not explitely set
    if(numberOfObjects == 0xFFFFFFFFFFFFFFFF) {
        numberOfObjects = buffer->numberOfObjects;
    }

    // check size
    if(offset + numberOfObjects > buffer->numberOfObjects) {
        return false;
    }

    if(buffer->useHostPtr == false
            && numberOfObjects != 0)
    {
        // write data into the buffer on the device
        const cl_int ret = m_queue.enqueueWriteBuffer(buffer->clBuffer,
                                                      CL_TRUE,
                                                      offset * objectSize,
                                                      numberOfObjects * objectSize,
                                                      buffer->data);
        if(ret != CL_SUCCESS) {
            return false;
        }
    }

    it->second.kernel.setArg((bufferId * 2) + 1, static_cast<cl_ulong>(numberOfObjects));

    return true;
}

/**
 * @brief run kernel with input
 *
 * @param data input-data for the run
 *
 * @return true, if successful, else false
 */
bool
Opencl::run(OpenClData &data,
            const std::string &kernelName)
{
    LOG_DEBUG("run kernel on OpenCL device");

    // precheck
    if(m_device.size() == 0)
    {
        LOG_ERROR("no device initialized.");
        return false;
    }

    // precheck
    if(validateWorkerGroupSize(data) == false) {
        return false;
    }

    // convert ranges
    const cl::NDRange globalRange = cl::NDRange(data.numberOfWg.x * data.threadsPerWg.x,
                                                data.numberOfWg.y * data.threadsPerWg.y,
                                                data.numberOfWg.z * data.threadsPerWg.z);
    const cl::NDRange localRange = cl::NDRange(data.threadsPerWg.x,
                                               data.threadsPerWg.y,
                                               data.threadsPerWg.z);

    std::map<std::string, KernelDef>::const_iterator it;
    it = m_kernel.find(kernelName);

    if(it == m_kernel.end())
    {
        LOG_ERROR("no kernel with name " + kernelName + " found");
        return false;
    }

    try
    {
        // launch kernel on the device
        const cl_int ret = m_queue.enqueueNDRangeKernel(it->second.kernel,
                                                        cl::NullRange,
                                                        globalRange,
                                                        localRange);
        if(ret != CL_SUCCESS) {
            return false;
        }
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
 * @brief copy data of all as output marked buffer from device to host
 *
 * @param data object with all data
 *
 * @return true, if successful, else false
 */
bool
Opencl::copyFromDevice(OpenClData &data)
{
    LOG_DEBUG("copy data from OpenCL device");

    // precheck
    if(m_device.size() == 0)
    {
        LOG_ERROR("no device initialized.");
        return false;
    }

    // get output back from device
    for(uint64_t i = 0; i < data.buffer.size(); i++)
    {
        if(data.buffer.at(i).isOutput)
        {
            // copy result back to host
            const cl_int ret = m_queue.enqueueReadBuffer(data.buffer[i].clBuffer,
                                                         CL_TRUE,
                                                         0,
                                                         data.buffer[i].numberOfBytes,
                                                         data.buffer[i].data);

            if(ret != CL_SUCCESS) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief close device, free buffer on device and delete all data from the data-object, which are
 *        not a null-pointer
 *
 * @param data object with all data related to the device, which will be cleared

 * @return true, if successful, else false
 */
bool
Opencl::closeDevice(OpenClData &data)
{
    LOG_DEBUG("close OpenCL device");

    // end queue
    const cl_int ret = m_queue.finish();
    if(ret != CL_SUCCESS) {
        return false;
    }

    // free allocated memory on the host
    for(uint64_t i = 0; i < data.buffer.size(); i++)
    {
        WorkerBuffer buffer = data.buffer.at(i);
        if(buffer.data != nullptr) {
            Kitsunemimi::alignedFree(buffer.data, buffer.numberOfBytes);
        }
    }

    // clear data and free memory on the device
    data.buffer.clear();

    // clear global variables
    m_device.clear();
    m_platform.clear();

    return true;
}

/**
 * @brief get size of the local memory on device
 *
 * @return size of local memory on device, or 0 if no device is initialized
 */
uint64_t
Opencl::getLocalMemorySize()
{
    // precheck
    if(m_device.size() == 0) {
        return 0;
    }

    // get information
    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get size of the global memory on device
 *
 * @return size of global memory on device, or 0 if no device is initialized
 */
uint64_t
Opencl::getGlobalMemorySize()
{
    // precheck
    if(m_device.size() == 0) {
        return 0;
    }

    // get information
    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get maximum memory size, which can be allocated at one time on device
 *
 * @return maximum at one time allocatable size, or 0 if no device is initialized
 */
uint64_t
Opencl::getMaxMemAllocSize()
{
    // precheck
    if(m_device.size() == 0) {
        return 0;
    }

    // get information
    cl_ulong size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);

    return size;
}

/**
 * @brief get maximum total number of work-items within a work-group
 *
 * @return maximum work-group size
 */
uint64_t
Opencl::getMaxWorkGroupSize()
{
    // precheck
    if(m_device.size() == 0) {
        return 0;
    }

    // get information
    size_t size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);

    return size;
}

/**
 * @brief get maximum size of all dimensions of work-items within a work-group
 *
 * @return worker-dimension object
 */
const WorkerDim
Opencl::getMaxWorkItemSize()
{
    // precheck
    if(m_device.size() == 0) {
        return WorkerDim();
    }

    // get information
    size_t size[3];
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size);

    // create result object
    const uint64_t dimension = getMaxWorkItemDimension();
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
 * @brief get maximaum dimension of items
 *
 * @return number of dimensions
 */
uint64_t
Opencl::getMaxWorkItemDimension()
{
    // precheck
    if(m_device.size() == 0) {
        return 0;
    }

    // get information
    cl_uint size = 0;
    m_device.at(0).getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &size);

    return size;
}

/**
 * @brief precheck to validate given worker-group size by comparing them with the maximum values
 *        defined by the device
 *
 * @param data data-object, which also contains the worker-dimensions
 *
 * @return true, if successful, else false
 */
bool
Opencl::validateWorkerGroupSize(const OpenClData &data)
{
    const uint64_t givenSize = data.threadsPerWg.x * data.threadsPerWg.y * data.threadsPerWg.z;
    const uint64_t maxSize = getMaxWorkGroupSize();

    // checko maximum size
    if(givenSize > maxSize)
    {
        LOG_ERROR("Size of the work-group is too big. The maximum allowed is "
                  + std::to_string(maxSize)
                  + ", but set was a total size of "
                  + std::to_string(givenSize));
        return false;
    }

    const WorkerDim maxDim = getMaxWorkItemSize();

    // check single dimensions
    if(data.threadsPerWg.x > maxDim.x)
    {
        LOG_ERROR("The x-dimension of the work-item size is only allowed to have a maximum of"
                  + std::to_string(maxDim.x));
        return false;
    }
    if(data.threadsPerWg.y > maxDim.y)
    {
        LOG_ERROR("The y-dimension of the work-item size is only allowed to have a maximum of"
                  + std::to_string(maxDim.y));
        return false;
    }
    if(data.threadsPerWg.z > maxDim.z)
    {
        LOG_ERROR("The z-dimension of the work-item size is only allowed to have a maximum of"
                  + std::to_string(maxDim.z));
        return false;
    }

    return true;
}

/**
 * @brief collect all available devices
 *
 * @param config object with config-parameter
 */
bool
Opencl::collectDevices()
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
        LOG_DEBUG("number of OpenCL devices: " + std::to_string(pldev.size()));

        // select devices within the platform
        std::vector<cl::Device>::const_iterator dev_it;
        for(dev_it = pldev.begin();
            dev_it != pldev.end();
            dev_it++)
        {
            // check if device is available
            if(dev_it->getInfo<CL_DEVICE_AVAILABLE>())
            {
                /*if(false)
                {
                    // check for double precision support
                    const std::string ext = dev_it->getInfo<CL_DEVICE_EXTENSIONS>();
                    if(ext.find("cl_khr_fp64") != std::string::npos
                        && ext.find("cl_amd_fp64") != std::string::npos)
                    {
                        m_devices.push_back(*dev_it);
                        m_context = cl::Context(m_devices);
                    }
                }
                else
                {*/
                    // add all devices
                    m_device.push_back(*dev_it);
                    m_context = cl::Context(m_device);
                //}
            }
        }
    }

    if(m_device.size() == 0)
    {
        LOG_ERROR("no device found.");
        return false;
    }

    return true;
}

/**
 * @brief build kernel-code
 *
 * @param config object with config-parameter
 *
 * @return true, if successful, else false
 */
bool
Opencl::build(KernelDef &def)
{
    // compile opencl program for found device.
    const std::pair<const char*, size_t> kernelCode = std::make_pair(def.kernelCode.c_str(),
                                                                     def.kernelCode.size());
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

    // create kernel
    def.kernel = cl::Kernel(program, def.id.c_str());

    return true;
}

}
}
