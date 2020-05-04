#include <iostream>
#include <vector>
#include <string>

#include <libKitsunemimiPersistence/logger/logger.h>

#include <libKitsunemimiOpencl/opencl.h>

int main()
{
    Kitsunemimi::Persistence::initConsoleLogger(true);

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
        "    size_t gloablId = get_global_id(0) + get_global_size(0) * get_global_id(1);\n"
        "    if (gloablId < n1)\n"
        "    {\n"
        "       temp[localId_x] = b[gloablId];\n"
        "       c[gloablId] = a[gloablId] + temp[localId_x] + temp[localId_x] + temp[localId_x] + temp[localId_x] + temp[localId_x];"
        "    }\n"
        "}\n";

    Kitsunemimi::Opencl::Opencl ocl;

    // create config-object
    Kitsunemimi::Opencl::OpenClConfig config;
    config.kernelCode = kernelCode;
    config.kernelName = "add";

    // create data-object
    Kitsunemimi::Opencl::OpenClData data;

    data.numberOfWg.x = N / 1024;
    data.numberOfWg.y = 2;
    data.threadsPerWg.x = 512;

    // init empty buffer
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float)));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float)));
    data.buffer.push_back(Kitsunemimi::Opencl::WorkerBuffer(N, sizeof(float), true));

    // convert pointer
    float* a = static_cast<float*>(data.buffer[0].data);
    float* b = static_cast<float*>(data.buffer[1].data);

    // write intput dat into buffer
    for(uint32_t i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    double duration = 0.0;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // run
    if(ocl.init(config))
    {
        std::cout<<std::endl;

        start = std::chrono::system_clock::now();
        ocl.copyToDevice(data);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout<<"copy to device: "<<std::to_string(duration / 1000.0)<<" us"<<std::endl;

        start = std::chrono::system_clock::now();
        ocl.run(data);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout<<"run: "<<std::to_string(duration / 1000.0)<<" us"<<std::endl;

        start = std::chrono::system_clock::now();
        ocl.copyFromDevice(data);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout<<"copy from device: "<<std::to_string(duration / 1000.0)<<" us"<<std::endl;
    }

    // check result
    float* outputValues = static_cast<float*>(data.buffer[2].data);
    // Should get '3' here.
    //std::cout << outputValues[42] << std::endl;
    /*for(uint64_t i = 0; i < N; i++)
    {
        std::cout<<outputValues[i]<<std::endl;
    }*/

    std::cout<<std::endl;
    std::cout<<"vendor: "<<ocl.getVendor()<<std::endl;
    std::cout<<"local-size: "<<ocl.getLocalMemorySize()<<std::endl;
    std::cout<<"global-size: "<<ocl.getGlobalMemorySize_total()<<std::endl;
    std::cout<<"global-available: "<<ocl.getGlobalMemorySize_free()<<std::endl;
    std::cout<<"max Mem alloc size: "<<ocl.getMaxMemAllocSize()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"max work-group-size: "<<ocl.getMaxWorkGroupSize()<<std::endl;
    std::cout<<"max work-item-dimension: "<<ocl.getMaxWorkItemDimension()<<std::endl;
    std::cout<<"max work-item-size (x): "<<ocl.getMaxWorkItemSize().x<<std::endl;
    std::cout<<"max work-item-size (y): "<<ocl.getMaxWorkItemSize().y<<std::endl;
    std::cout<<"max work-item-size (z): "<<ocl.getMaxWorkItemSize().z<<std::endl;

}
