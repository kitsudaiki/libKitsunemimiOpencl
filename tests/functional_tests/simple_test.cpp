/**
 * @file        simple_test.cpp
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

#include "simple_test.h"

#include <libKitsunemimiOpencl/gpu_interface.h>
#include <libKitsunemimiOpencl/gpu_handler.h>

namespace Kitsunemimi
{
namespace Opencl
{

SimpleTest::SimpleTest()
    : Kitsunemimi::CompareTestHelper("SimpleTest")
{
    simple_test();
}

void
SimpleTest::simple_test()
{
    const size_t testSize = 1 << 27;

    // example kernel for task: c = a + b.
    const std::string kernelCode =
        "__kernel void add(\n"
        "       __global const float* a,\n"
        "       ulong n1,\n"
        "       __global const float* b,\n"
        "       ulong n2,\n"
        "       __global float* c,\n"
        "       ulong out,\n"
        "       __local uchar* localMemory,\n"
        "       const ulong localMemorySize\n"
        "       )\n"
        "{\n"
        "    __local float temp[512];\n"
        "    size_t globalId_x = get_global_id(0);\n"
        "    int localId_x = get_local_id(0);\n"
        "    size_t globalSize_x = get_global_size(0);\n"
        "    size_t globalSize_y = get_global_size(1);\n"
        "    \n"
        "    size_t globalId = get_global_id(0) + get_global_size(0) * get_global_id(1);\n"
        "    if (globalId < n1)\n"
        "    {\n"
        "       temp[localId_x] = b[globalId];\n"
        "       c[globalId] = a[globalId] + temp[localId_x];"
        "    }\n"
        "}\n";

    Kitsunemimi::Opencl::GpuHandler oclHandler;

    TEST_NOT_EQUAL(oclHandler.m_interfaces.size(), 0);

    Kitsunemimi::Opencl::GpuInterface* ocl = oclHandler.m_interfaces.at(0);

    // create data-object
    Kitsunemimi::Opencl::OpenClData data;

    data.numberOfWg.x = testSize / 512;
    data.numberOfWg.y = 2;
    data.threadsPerWg.x = 256;

    // init empty buffer
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(testSize, sizeof(float), false, true));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(testSize, sizeof(float), false, true));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(testSize, sizeof(float), true, true));

    // convert pointer
    float* a = static_cast<float*>(data.buffer[0].data);
    float* b = static_cast<float*>(data.buffer[1].data);

    // write intput dat into buffer
    for(uint32_t i = 0; i < testSize; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // run
    TEST_EQUAL(ocl->initCopyToDevice(data), true);
    TEST_EQUAL(ocl->addKernel("add", kernelCode), true);
    TEST_EQUAL(ocl->bindKernelToBuffer("add", 0, data), true);
    TEST_EQUAL(ocl->bindKernelToBuffer("add", 1, data), true);
    TEST_EQUAL(ocl->bindKernelToBuffer("add", 2, data), true);
    TEST_EQUAL(ocl->setLocalMemory("add", 256*256), true);
    TEST_EQUAL(ocl->run(data, "add"), true);
    TEST_EQUAL(ocl->copyFromDevice(data), true);

    // check result
    float* outputValues = static_cast<float*>(data.buffer[2].data);
    TEST_EQUAL(outputValues[42], 3.0f);;

    // update data on host
    for(uint32_t i = 0; i < testSize; i++)
    {
        a[i] = 5.0f;
    }

    // update data on device
    TEST_EQUAL(ocl->updateBufferOnDevice("add", 0), true);

    // second run
    TEST_EQUAL(ocl->run(data, "add"), true);
    // copy new output back
    TEST_EQUAL(ocl->copyFromDevice(data), true);

    // check new result
    outputValues = static_cast<float*>(data.buffer[2].data);
    TEST_EQUAL(outputValues[42], 7.0f);

    // test memory getter
    TEST_NOT_EQUAL(ocl->getLocalMemorySize(), 0);
    TEST_NOT_EQUAL(ocl->getGlobalMemorySize(), 0);
    TEST_NOT_EQUAL(ocl->getMaxMemAllocSize(), 0);

    // test work group getter
    TEST_NOT_EQUAL(ocl->getMaxWorkGroupSize(), 0);
    TEST_NOT_EQUAL(ocl->getMaxWorkItemDimension(), 0);
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().x, 0);
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().y, 0);
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().z, 0);

    // test close
    TEST_EQUAL(ocl->closeDevice(data), true);
}

}
}
