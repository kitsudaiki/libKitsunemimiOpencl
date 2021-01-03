#include <libKitsunemimiOpencl/gpu_data.h>

namespace Kitsunemimi
{
namespace Opencl
{

GpuData::GpuData()
{

}

bool
GpuData::addBuffer(const std::string &name, const WorkerBuffer &buffer)
{
    if(containsBuffer(name)) {
        return false;
    }

    m_buffer.insert(std::make_pair(name, buffer));

    return true;
}

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

}
}
