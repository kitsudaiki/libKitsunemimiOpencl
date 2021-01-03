/**
 * @file        gpu_data.h
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

#ifndef GPU_DATA_H
#define GPU_DATA_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace Kitsunemimi
{
namespace Opencl
{
class GpuInterface;

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

class GpuData
{
public:
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;

    GpuData();

    bool addBuffer(const std::string &name,
                   const WorkerBuffer &buffer);
    bool containsBuffer(const std::string &name);
    void* getBufferData(const std::string &name);

private:
    friend GpuInterface;

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
    };

    std::map<std::string, WorkerBuffer> m_buffer;
    std::map<std::string, KernelDef> m_kernel;

    WorkerBuffer* getBuffer(const std::string &name);

    bool containsKernel(const std::string &name);
    KernelDef* getKernel(const std::string &name);
};

}
}

#endif // GPU_DATA_H