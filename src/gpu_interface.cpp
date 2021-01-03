/**
 * @file        gpu_interface.cpp
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

#include <libKitsunemimiOpencl/gpu_interface.h>

#include <libKitsunemimiPersistence/logger/logger.h>

namespace Kitsunemimi
{
namespace Opencl
{

/**
 * @brief constructor
 *
 * @param device opencl-device
 */
GpuInterface::GpuInterface(const cl::Device &device)
{
    LOG_DEBUG("created new gpu-interface for OpenCL device: " + device.getInfo<CL_DEVICE_NAME>());

    m_device = device;
    m_context = cl::Context(m_device);
    m_queue = cl::CommandQueue(m_context, m_device);
}

/**
 * @brief destructor to close at least the device-connection
 */
GpuInterface::~GpuInterface()
{
    GpuData emptyData;
    closeDevice(emptyData);
}

/**
 * @brief copy data from host to device
 *
 * @param data object with all data
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::initCopyToDevice(GpuData &data)
{
    LOG_DEBUG("initial data transfer to OpenCL device");

    // send input to device
    std::map<std::string, GpuData::WorkerBuffer>::iterator it;
    for(it = data.m_buffer.begin();
        it != data.m_buffer.end();
        it++)
    {

        GpuData::WorkerBuffer* buffer = &it->second;

        LOG_DEBUG("copy data to device: "
                  + std::to_string(buffer->numberOfBytes)
                  + " Bytes");

        // check buffer
        if(buffer->numberOfBytes == 0
                || buffer->numberOfObjects == 0
                || buffer->data == nullptr)
        {
            LOG_ERROR("failed to copy data to device, because buffer with name '"
                      + it->first
                      + "' has size 0 or is not initialized.");
            return false;
        }

        // create flag for memory handling
        cl_mem_flags flags = 0;
        if(buffer->isOutput)
        {
            flags = CL_MEM_READ_WRITE;
            if(buffer->useHostPtr) {
                flags = flags | CL_MEM_USE_HOST_PTR;
            } else {
                flags = flags | CL_MEM_COPY_HOST_PTR;
            }
        }
        else
        {
            if(buffer->useHostPtr) {
                flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
            } else {
                flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            }
        }

        // send data or reference to device
        it->second.clBuffer = cl::Buffer(m_context,
                                         flags,
                                         buffer->numberOfBytes,
                                         buffer->data);
    }

    return true;
}

/**
 * @brief add kernel to device
 *
 * @param data object with all data
 * @param kernelName name of the kernel
 * @param kernelCode kernel source-code as string
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::addKernel(GpuData &data,
                        const std::string &kernelName,
                        const std::string &kernelCode)
{
    LOG_DEBUG("add kernel with id: " + kernelName);

    // TODO: check if already registerd
    GpuData::KernelDef def;
    def.id = kernelName;
    def.kernelCode = kernelCode;

    try
    {
        const bool buildResult = build(def);
        if(buildResult == false) {
            return false;
        }

        data.m_kernel.insert(std::make_pair(kernelName, def));
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
 * @brief bind a buffer to a kernel
 *
 * @param data data-object with the buffer to bind
 * @param kernelName, name of the kernel, which should be used
 * @param bufferName name of buffer, which should be bind to the kernel
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::bindKernelToBuffer(GpuData &data,
                                 const std::string &kernelName,
                                 const std::string &bufferName)
{
    LOG_DEBUG("bind buffer with name '" + bufferName + "' kernel with name: '" + kernelName + "'");

    // get kernel-data
    if(data.containsKernel(kernelName) == false)
    {
        LOG_ERROR("no kernel with name '" + kernelName + "' found");
        return false;
    }

    // check if buffer-name exist
    if(data.containsBuffer(bufferName) == false)
    {
        LOG_ERROR("no buffer with name '" + bufferName + "' found");
        return false;
    }

    GpuData::KernelDef* def = data.getKernel(kernelName);
    GpuData::WorkerBuffer* buffer = &data.m_buffer[bufferName];

    // register arguments in opencl
    const uint32_t argNumber = static_cast<uint32_t>(def->bufferLinks.size() * 2);
    LOG_DEBUG("bind buffer with name "
              + bufferName
              + "' to argument number "
              + std::to_string(argNumber));
    def->kernel.setArg(argNumber, buffer->clBuffer);
    LOG_DEBUG("bind size value of buffer with name '"
              + bufferName
              + "' to argument number "
              + std::to_string(argNumber + 1));
    def->kernel.setArg(argNumber + 1, static_cast<cl_ulong>(buffer->numberOfObjects));

    // create buffer-link for later access
    GpuData::BufferLink link;
    link.buffer = buffer;
    link.bindedId = argNumber;
    def->bufferLinks.insert(std::make_pair(bufferName, link));

    return true;
}

/**
 * @brief Opencl::setLocalMemory
 *
 * @param data object with all data
 * @param kernelName, name of the kernel, which should be executed
 * @param localMemorySize size of the local mamory
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::setLocalMemory(GpuData &data,
                             const std::string &kernelName,
                             const uint32_t localMemorySize)
{
    // get kernel-data
    if(data.containsKernel(kernelName) == false)
    {
        LOG_ERROR("no kernel with name '" + kernelName + "' found");
        return false;
    }

    GpuData::KernelDef* def = data.getKernel(kernelName);
    const uint32_t argNumber = static_cast<uint32_t>(def->bufferLinks.size()) * 2;
    // set arguments
    def->kernel.setArg(argNumber, localMemorySize, nullptr);
    def->kernel.setArg(argNumber + 1, static_cast<cl_ulong>(localMemorySize));

    return true;
}


/**
 * @brief update data inside the buffer on the device
 *
 * @param data object with all data
 * @param bufferName name of the buffer in the kernel
 * @param numberOfObjects number of objects to copy
 * @param offset offset in buffer on device
 *
 * @return false, if copy failed of buffer is output-buffer, else true
 */
bool
GpuInterface::updateBufferOnDevice(GpuData &data,
                                   const std::string &bufferName,
                                   uint64_t numberOfObjects,
                                   const uint64_t offset)
{
    LOG_DEBUG("update buffer on OpenCL device");

    // check id
    if(data.containsBuffer(bufferName) == false)
    {
        LOG_ERROR("no buffer with name '" + bufferName + "' found");
        return false;
    }

    GpuData::WorkerBuffer* buffer = data.getBuffer(bufferName);
    const uint64_t objectSize = buffer->numberOfBytes / buffer->numberOfObjects;

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

    // update buffer
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

    return true;
}

/**
 * @brief run kernel with input
 *
 * @param data input-data for the run
 * @param kernelName, name of the kernel, which should be executed
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::run(GpuData &data,
                  const std::string &kernelName)
{
    //LOG_DEBUG("run kernel on OpenCL device");

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

    // get kernel-data
    if(data.containsKernel(kernelName) == false)
    {
        LOG_ERROR("no kernel with name '" + kernelName + "' found");
        return false;
    }

    try
    {
        // launch kernel on the device
        const cl_int ret = m_queue.enqueueNDRangeKernel(data.getKernel(kernelName)->kernel,
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
GpuInterface::copyFromDevice(GpuData &data)
{
    // get output back from device
    std::map<std::string, GpuData::WorkerBuffer>::iterator it;
    for(it = data.m_buffer.begin();
        it != data.m_buffer.end();
        it++)
    {
        if(it->second.isOutput)
        {
            // copy result back to host
            const cl_int ret = m_queue.enqueueReadBuffer(it->second.clBuffer,
                                                         CL_TRUE,
                                                         0,
                                                         it->second.numberOfBytes,
                                                         it->second.data);

            if(ret != CL_SUCCESS) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief GpuInterface::getDeviceName
 * @return
 */
const std::string
GpuInterface::getDeviceName()
{
    return m_device.getInfo<CL_DEVICE_NAME>();
}

/**
 * @brief close device, free buffer on device and delete all data from the data-object, which are
 *        not a null-pointer
 *
 * @param data object with all data related to the device, which will be cleared

 * @return true, if successful, else false
 */
bool
GpuInterface::closeDevice(GpuData &data)
{
    LOG_DEBUG("close OpenCL device");

    // end queue
    const cl_int ret = m_queue.finish();
    if(ret != CL_SUCCESS) {
        return false;
    }

    // free allocated memory on the host
    std::map<std::string, GpuData::WorkerBuffer>::iterator it;
    for(it = data.m_buffer.begin();
        it != data.m_buffer.end();
        it++)
    {
        if(it->second.data != nullptr
                && it->second.allowBufferDeleteAfterClose)
        {
            Kitsunemimi::alignedFree(it->second.data, it->second.numberOfBytes);
        }
    }

    // clear data and free memory on the device
    data.m_buffer.clear();

    return true;
}

/**
 * @brief get size of the local memory on device
 *
 * @return size of local memory on device, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getLocalMemorySize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get size of the global memory on device
 *
 * @return size of global memory on device, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getGlobalMemorySize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get maximum memory size, which can be allocated at one time on device
 *
 * @return maximum at one time allocatable size, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getMaxMemAllocSize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);

    return size;
}

/**
 * @brief get maximum total number of work-items within a work-group
 *
 * @return maximum work-group size
 */
uint64_t
GpuInterface::getMaxWorkGroupSize()
{
    // get information
    size_t size = 0;
    m_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);

    return size;
}

/**
 * @brief get maximum size of all dimensions of work-items within a work-group
 *
 * @return worker-dimension object
 */
const WorkerDim
GpuInterface::getMaxWorkItemSize()
{
    // get information
    size_t size[3];
    m_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size);

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
GpuInterface::getMaxWorkItemDimension()
{
    // get information
    cl_uint size = 0;
    m_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &size);

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
GpuInterface::validateWorkerGroupSize(const GpuData &data)
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
 * @brief build kernel-code
 *
 * @param def kernel struct object with kernel-name and kernel-code
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::build(GpuData::KernelDef &data)
{
    // compile opencl program for found device.
    const std::pair<const char*, size_t> kernelCode = std::make_pair(data.kernelCode.c_str(),
                                                                     data.kernelCode.size());
    const cl::Program::Sources source = cl::Program::Sources(1, kernelCode);
    cl::Program program(m_context, source);

    try
    {
        std::vector<cl::Device> devices = {m_device};
        program.build(devices);
    }
    catch(const cl::Error&)
    {
        LOG_ERROR("OpenCL compilation error\n    "
                  + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device));
        return false;
    }

    // create kernel
    data.kernel = cl::Kernel(program, data.id.c_str());

    return true;
}

}
}
