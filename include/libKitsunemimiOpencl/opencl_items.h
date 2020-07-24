/**
 * @file        opencl_items.h
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

#ifndef OPENCL_ITEMS_H
#define OPENCL_ITEMS_H

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include <CL/cl.hpp>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

namespace Kitsunemimi
{
namespace Opencl
{

enum DeviceType
{
    GPU_TYPE,
    CPU_TYPE,
    ALL_TYPE
};

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
    uint32_t argumentId = 0;
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

struct OpenClConfig
{
    std::map<std::string, std::string> kernelDefinition;

    DeviceType type = GPU_TYPE;
    uint32_t maxNumberOfDevice = 1; // 0 = all
    bool requiresDoublePrecision = false;
};

struct OpenClData
{
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;

    uint64_t localMemorySize = 0;
    std::vector<WorkerBuffer> buffer;
};

}
}

#endif // OPENCL_ITEMS_H
