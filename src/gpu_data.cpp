/**
 * @file        gpu_data.cpp
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

#include <libKitsunemimiOpencl/gpu_data.h>

namespace Kitsunemimi
{
namespace Opencl
{

GpuData::GpuData() {}

/**
 * @brief GpuData::addBuffer
 * @param name
 * @param buffer
 * @return
 */
bool
GpuData::addBuffer(const std::string &name,
                   const uint64_t numberOfObjects,
                   const uint64_t objectSize,
                   const bool isOutput,
                   const bool useHostPtr)
{
    if(containsBuffer(name)) {
        return false;
    }

    WorkerBuffer newBuffer(numberOfObjects, objectSize, isOutput, useHostPtr);
    m_buffer.insert(std::make_pair(name, newBuffer));

    return true;
}

/**
 * @brief GpuData::getBuffer
 * @param name
 * @return
 */
GpuData::WorkerBuffer*
GpuData::getBuffer(const std::string &name)
{
    std::map<std::string, WorkerBuffer>::iterator it;
    it = m_buffer.find(name);
    if(it != m_buffer.end()) {
        return &it->second;
    }

    return nullptr;
}

/**
 * @brief GpuData::containsBuffer
 * @param name
 * @return
 */
bool
GpuData::containsBuffer(const std::string &name)
{
    std::map<std::string, WorkerBuffer>::const_iterator it;
    it = m_buffer.find(name);
    if(it != m_buffer.end()) {
        return true;
    }

    return false;
}

/**
 * @brief GpuData::getData
 * @param name
 * @return
 */
void*
GpuData::getBufferData(const std::string &name)
{
    std::map<std::string, WorkerBuffer>::iterator it;
    it = m_buffer.find(name);
    if(it != m_buffer.end()) {
        return it->second.data;
    }

    return nullptr;
}

/**
 * @brief GpuData::containsKernel
 * @param name
 * @return
 */
bool
GpuData::containsKernel(const std::string &name)
{
    std::map<std::string, KernelDef>::const_iterator it;
    it = m_kernel.find(name);
    if(it != m_kernel.end()) {
        return true;
    }

    return false;
}

/**
 * @brief GpuData::getKernel
 * @param name
 * @return
 */
GpuData::KernelDef*
GpuData::getKernel(const std::string &name)
{
    std::map<std::string, KernelDef>::iterator it;
    it = m_kernel.find(name);
    if(it != m_kernel.end()) {
        return &it->second;
    }

    return nullptr;
}

}
}
