/**
 * @file        opencl.h
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

#include <libKitsunemimiOpencl/opencl_items.h>

namespace Kitsunemimi
{
namespace Opencl
{

class Opencl
{
public:
    Opencl();
    ~Opencl();

    // device-interaction
    bool initDevice(const OpenClConfig &config);
    bool initCopyToDevice(OpenClData &data);
    bool updateBufferOnDevice(WorkerBuffer &buffer,
                              uint64_t numberOfObjects = 0xFFFFFFFFFFFFFFFF,
                              const uint64_t offset = 0);
    bool run(OpenClData &data, const std::string &kernelName);
    bool copyFromDevice(OpenClData &data);
    bool closeDevice(OpenClData &data);

    // getter for memory information
    uint64_t getLocalMemorySize();
    uint64_t getGlobalMemorySize();
    uint64_t getMaxMemAllocSize();

    // getter for work-group information
    uint64_t getMaxWorkGroupSize();
    WorkerDim getMaxWorkItemSize();
    uint64_t getMaxWorkItemDimension();

    // opencl objects
    // I left these public for the case, that there have to be some specific operations have to be
    // performed, which are not possible or available with the generic functions of this library.
    std::vector<cl::Platform> m_platform;
    std::vector<cl::Device> m_device;
    cl::Context m_context;
    std::map<std::string, cl::Kernel> m_kernel;
    cl::CommandQueue m_queue;
    uint32_t m_argCounter = 0;

private:
    bool validateWorkerGroupSize(const OpenClData &data);
    void collectDevices(const OpenClConfig &config);
    bool build(const OpenClConfig &config);
};

}
}

#endif // OPENCL_H
