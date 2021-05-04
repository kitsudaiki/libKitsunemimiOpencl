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
    : Kitsunemimi::SpeedTestHelper()
{
    m_copyToDeviceTimeSlot.unitName = "ms";
    m_copyToDeviceTimeSlot.name = "copy to device";

    m_initKernelTimeSlot.unitName = "ms";
    m_initKernelTimeSlot.name = "init kernel";

    m_runTimeSlot.unitName = "ms";
    m_runTimeSlot.name = "run test";

    m_updateTimeSlot.unitName = "ms";
    m_updateTimeSlot.name = "update data on device";

    m_copyToHostTimeSlot.unitName = "ms";
    m_copyToHostTimeSlot.name = "copy to host";

    m_cleanupTimeSlot.unitName = "ms";
    m_cleanupTimeSlot.name = "cleanup";

    m_oclHandler = new Kitsunemimi::Opencl::GpuHandler();
    assert(m_oclHandler->m_interfaces.size() != 0);

    chooseDevice();

    for(uint32_t i = 0; i < 10; i++)
    {
        std::cout<<"run cycle "<<(i + 1)<<std::endl;

        simple_test();

        m_copyToDeviceTimeSlot.values.push_back(
                    m_copyToDeviceTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_initKernelTimeSlot.values.push_back(
                    m_initKernelTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_runTimeSlot.values.push_back(
                    m_runTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_updateTimeSlot.values.push_back(
                    m_updateTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_copyToHostTimeSlot.values.push_back(
                    m_copyToHostTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_cleanupTimeSlot.values.push_back(
                    m_cleanupTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
    }

    addToResult(m_copyToDeviceTimeSlot);
    addToResult(m_initKernelTimeSlot);
    addToResult(m_runTimeSlot);
    addToResult(m_updateTimeSlot);
    addToResult(m_copyToHostTimeSlot);
    addToResult(m_cleanupTimeSlot);

    printResult();
}

void
SimpleTest::simple_test()
{
    const size_t testSize = 1 << 28;

    // example kernel for task: c = a + b.
    const std::string kernelCode =
        "__kernel void add(\n"
        "       __global const float* a,\n"
        "       ulong n1,\n"
        "       __global const float* b,\n"
        "       ulong n2,\n"
        "       __global float* c,\n"
        "       ulong out\n"
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
    Kitsunemimi::Opencl::GpuInterface* ocl = oclHandler.m_interfaces.at(m_id);

    // create data-object
    Kitsunemimi::Opencl::GpuData data;

    data.numberOfWg.x = testSize / 512;
    data.numberOfWg.y = 2;
    data.threadsPerWg.x = 256;

    // init empty buffer
    data.addBuffer("x", testSize, sizeof(float), false);
    data.addBuffer("y", testSize, sizeof(float), false);
    data.addBuffer("z", testSize, sizeof(float), false);

    // convert pointer
    float* a = static_cast<float*>(data.getBufferData("x"));
    float* b = static_cast<float*>(data.getBufferData("y"));

    // write intput dat into buffer
    for(uint32_t i = 0; i < testSize; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // copy to device
    m_copyToDeviceTimeSlot.startTimer();
    assert(ocl->initCopyToDevice(data));
    m_copyToDeviceTimeSlot.stopTimer();

    m_initKernelTimeSlot.startTimer();
    assert(ocl->addKernel(data, "add", kernelCode));
    assert(ocl->bindKernelToBuffer(data, "add", "x"));
    assert(ocl->bindKernelToBuffer(data, "add", "y"));
    assert(ocl->bindKernelToBuffer(data, "add", "z"));
    m_initKernelTimeSlot.stopTimer();

    // run
    m_runTimeSlot.startTimer();
    assert(ocl->run(data, "add"));
    m_runTimeSlot.stopTimer();

    // copy output back
    m_copyToHostTimeSlot.startTimer();
    assert(ocl->copyFromDevice(data, "z"));
    m_copyToHostTimeSlot.stopTimer();

    // update data on host
    for(uint32_t i = 0; i < testSize; i++)
    {
        a[i] = 5.0f;
    }

    // update data on device
    m_updateTimeSlot.startTimer();
    assert(ocl->updateBufferOnDevice(data, "x"));
    m_updateTimeSlot.stopTimer();

    // clear device
    m_cleanupTimeSlot.startTimer();
    assert(ocl->closeDevice(data));
    m_cleanupTimeSlot.stopTimer();
}

void
SimpleTest::chooseDevice()
{
    std::cout<<"found devices:"<<std::endl;
    for(uint32_t i = 0; i < m_oclHandler->m_interfaces.size(); i++) {
        std::cout<<"    "<<i<<": "<<m_oclHandler->m_interfaces.at(i)->getDeviceName()<<std::endl;
    }

    while(m_id >= m_oclHandler->m_interfaces.size())
    {
        std::cout<<"wait for input: ";
        std::cin>>m_id;
    }
}

}
}
