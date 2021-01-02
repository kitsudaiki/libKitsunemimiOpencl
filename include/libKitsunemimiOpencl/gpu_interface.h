/**
 * @file        gpu_interface.h
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

#ifndef OPENCL_H
#define OPENCL_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <libKitsunemimiCommon/buffer/data_buffer.h>

namespace Kitsunemimi
{
namespace Opencl
{

struct WorkerDim
{
    uint64_t x = 1;
    uint64_t y = 1;
    uint64_t z = 1;
};

struct WorkerBuffer
{
    void* data = nullptr;
    uint64_t numberOfBytes = 0;
    uint64_t numberOfObjects = 0;
    bool isOutput = false;
    bool useHostPtr = false;
    cl::Buffer clBuffer;

    WorkerBuffer() {}

    WorkerBuffer(const uint64_t numberOfObjects,
                 const uint64_t objectSize,
                 const bool isOutput = false,
                 const bool useHostPtr = false)
    {
        this->numberOfBytes = numberOfObjects * objectSize;
        this->data = Kitsunemimi::alignedMalloc(4096, numberOfBytes);
        this->numberOfObjects = numberOfObjects;
        this->isOutput = isOutput;
        this->useHostPtr = useHostPtr;
    }
};

struct OpenClData
{
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;
    std::map<std::string, WorkerBuffer> buffer;

    bool containsBuffer(const std::string &name)
    {
        std::map<std::string, WorkerBuffer>::const_iterator it;
        it = buffer.find(name);
        if(it != buffer.end()) {
            return true;
        }

        return false;
    }
};


class GpuInterface
{
public:
    GpuInterface(const cl::Device &device);
    ~GpuInterface();

    // initializing
    bool initCopyToDevice(OpenClData &data);

    bool addKernel(const std::string &kernelName,
                   const std::string &kernelCode);
    bool bindKernelToBuffer(const std::string &kernelName,
                            const std::string &bufferName,
                            OpenClData &data);
    bool setLocalMemory(const std::string &kernelName,
                        const uint32_t localMemorySize);

    bool closeDevice(OpenClData &data);

    // runtime
    bool updateBufferOnDevice(const std::string &kernelName,
                              const std::string &bufferName,
                              uint64_t numberOfObjects = 0xFFFFFFFFFFFFFFFF,
                              const uint64_t offset = 0);
    bool run(const std::string &kernelName,
             OpenClData &data);
    bool copyFromDevice(OpenClData &data);

    // common getter
    const std::string getDeviceName();

    // getter for memory information
    uint64_t getLocalMemorySize();
    uint64_t getGlobalMemorySize();
    uint64_t getMaxMemAllocSize();

    // getter for work-group information
    uint64_t getMaxWorkGroupSize();
    const WorkerDim getMaxWorkItemSize();
    uint64_t getMaxWorkItemDimension();

private:
    struct BufferLink
    {
        WorkerBuffer* buffer = nullptr;
        uint32_t bindedId = 0;
        uint8_t padding[4];
    };

    struct KernelDef
    {
        std::string id = "";
        std::string kernelCode = "";
        cl::Kernel kernel;
        std::map<std::string, BufferLink> bufferLinks;
        uint32_t localBufferSize = 0;
        uint32_t argumentCounter = 0;

        bool containsBuffer(const std::string &name)
        {
            std::map<std::string, BufferLink>::const_iterator it;
            it = bufferLinks.find(name);
            if(it != bufferLinks.end()) {
                return true;
            }

            return false;
        }
    };

    cl::Device m_device;
    std::map<std::string, KernelDef> m_kernel;

    cl::Context m_context;
    cl::CommandQueue m_queue;

    bool validateWorkerGroupSize(const OpenClData &data);
    bool build(KernelDef &def);
};

}
}

#endif // OPENCL_H
