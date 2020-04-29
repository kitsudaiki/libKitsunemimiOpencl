#include <iostream>
#include <vector>
#include <string>

#include <libKitsunemimiOpencl/opencl.h>

int main()
{
    // Compute c = a + b.
    const std::string kernelCode =
        "kernel void add(\n"
        "       ulong n,\n"
        "       global const float *a,\n"
        "       global const float *b,\n"
        "       global float *c\n"
        "       )\n"
        "{\n"
        "    size_t i = get_global_id(0);\n"
        "    if (i < n) {\n"
        "       c[i] = a[i] + b[i];\n"
        "    }\n"
        "}\n";

    Kitsunemimi::Opencl::Opencl ocl;
    if(ocl.init()) {
        ocl.run(kernelCode);
    }
}
