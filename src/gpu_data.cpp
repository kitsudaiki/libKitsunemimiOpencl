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
GpuData::addBuffer(const std::string &name, const WorkerBuffer &buffer)
{
    if(containsBuffer(name)) {
        return false;
    }

    m_buffer.insert(std::make_pair(name, buffer));

    return true;
}

/**
 * @brief GpuData::getBuffer
 * @param name
 * @return
 */
WorkerBuffer*
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
