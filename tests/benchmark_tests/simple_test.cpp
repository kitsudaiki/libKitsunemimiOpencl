#include "simple_test.h"

#include <libKitsunemimiOpencl/opencl.h>

namespace Kitsunemimi
{
namespace Opencl
{

SimpleTest::SimpleTest()
    : Kitsunemimi::SpeedTestHelper()
{
    m_initTimeSlot.unitName = "ms";
    m_initTimeSlot.name = "init";

    m_copyToDeviceTimeSlot.unitName = "ms";
    m_copyToDeviceTimeSlot.name = "copy to device";

    m_runTimeSlot.unitName = "ms";
    m_runTimeSlot.name = "run test";

    m_updateTimeSlot.unitName = "ms";
    m_updateTimeSlot.name = "update data on device";

    m_copyToHostTimeSlot.unitName = "ms";
    m_copyToHostTimeSlot.name = "copy to host";

    m_cleanupTimeSlot.unitName = "ms";
    m_cleanupTimeSlot.name = "cleanup";

    for(uint32_t i = 0; i < 2; i++)
    {
        simple_test();

        m_initTimeSlot.values.push_back(
                    m_initTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_copyToDeviceTimeSlot.values.push_back(
                    m_copyToDeviceTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_runTimeSlot.values.push_back(
                    m_runTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_updateTimeSlot.values.push_back(
                    m_updateTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_copyToHostTimeSlot.values.push_back(
                    m_copyToHostTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
        m_cleanupTimeSlot.values.push_back(
                    m_cleanupTimeSlot.getDuration(MICRO_SECONDS) / 1000.0);
    }

    addToResult(m_initTimeSlot);
    addToResult(m_copyToDeviceTimeSlot);
    addToResult(m_runTimeSlot);
    addToResult(m_updateTimeSlot);
    addToResult(m_copyToHostTimeSlot);
    addToResult(m_cleanupTimeSlot);

    printResult();
}

void
SimpleTest::simple_test()
{
    const size_t N = 1 << 27;

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

    Kitsunemimi::Opencl::Opencl ocl;

    // create config-object
    Kitsunemimi::Opencl::OpenClConfig config;
    config.kernelCode = kernelCode;
    config.kernelName = "add";

    // create data-object
    Kitsunemimi::Opencl::OpenClData data;

    data.numberOfWg.x = N / 512;
    data.numberOfWg.y = 2;
    data.threadsPerWg.x = 256;

    // init empty buffer
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float), false, true));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float), false, true));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float), true, true));

    // convert pointer
    float* a = static_cast<float*>(data.buffer[0].data);
    float* b = static_cast<float*>(data.buffer[1].data);

    // write intput dat into buffer
    for(uint32_t i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // init
    m_initTimeSlot.startTimer();
    assert(ocl.initDevice(config));
    m_initTimeSlot.stopTimer();

    // copy to device
    m_copyToDeviceTimeSlot.startTimer();
    assert(ocl.initCopyToDevice(data));
    m_copyToDeviceTimeSlot.stopTimer();

    // run
    m_runTimeSlot.startTimer();
    assert(ocl.run(data));
    m_runTimeSlot.stopTimer();

    // copy output back
    m_copyToHostTimeSlot.startTimer();
    assert(ocl.copyFromDevice(data));
    m_copyToHostTimeSlot.stopTimer();

    m_cleanupTimeSlot.startTimer();
    m_cleanupTimeSlot.stopTimer();

    // update data on host
    for(uint32_t i = 0; i < N; i++)
    {
        a[i] = 5.0f;
    }

    // update data on device
    m_updateTimeSlot.startTimer();
    assert(ocl.updateBufferOnDevice(data.buffer[0]));
    m_updateTimeSlot.stopTimer();
}

}
}
