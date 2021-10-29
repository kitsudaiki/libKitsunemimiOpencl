/**
 * @file        gpu_handler.cpp
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

#include <libKitsunemimiOpencl/gpu_handler.h>

#include <libKitsunemimiOpencl/gpu_interface.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
namespace Opencl
{

GpuHandler::GpuHandler()
{
    initDevice();
}

/**
 * @brief initialize opencl
 *
 * @param config object with config-parameter
 *
 * @return true, if creation was successful, else false
 */
bool
GpuHandler::initDevice()
{
    LOG_DEBUG("initialize OpenCL device");

    try
    {
        // get all available opencl platforms
        cl::Platform::get(&m_platform);
        if(m_platform.empty())
        {
            ErrorContainer error;
            error.errorMessage = "No OpenCL platforms found.";
            error.possibleSolution = "install a graphics card.";
            LOG_ERROR(error);
            return false;
        }

        LOG_DEBUG("number of OpenCL platforms: " + std::to_string(m_platform.size()));

        collectDevices();

        return true;
    }
    catch(const cl::Error &err)
    {
        ErrorContainer error;
        error.errorMessage = "OpenCL error: "
                             + std::string(err.what())
                             + "("
                             + std::to_string(err.err())
                             + ")";
        LOG_ERROR(error);
        return false;
    }
}

/**
 * @brief collect all available devices
 *
 * @param config object with config-parameter
 */
void
GpuHandler::collectDevices()
{
    // get available platforms
    std::vector<cl::Platform>::const_iterator plat_it;
    for(plat_it = m_platform.begin();
        plat_it != m_platform.end();
        plat_it++)
    {
        // get available devices of the selected platform
        std::vector<cl::Device> pldev;
        plat_it->getDevices(CL_DEVICE_TYPE_ALL, &pldev);
        LOG_DEBUG("number of OpenCL devices: " + std::to_string(pldev.size()));

        // select devices within the platform
        std::vector<cl::Device>::const_iterator dev_it;
        for(dev_it = pldev.begin();
            dev_it != pldev.end();
            dev_it++)
        {
            // check if device is available
            if(dev_it->getInfo<CL_DEVICE_AVAILABLE>())
            {
                /*if(false)
                {
                    // check for double precision support
                    const std::string ext = dev_it->getInfo<CL_DEVICE_EXTENSIONS>();
                    if(ext.find("cl_khr_fp64") != std::string::npos
                        && ext.find("cl_amd_fp64") != std::string::npos)
                    {
                        m_devices.push_back(*dev_it);
                        m_context = cl::Context(m_devices);
                    }
                }*/

                m_interfaces.push_back(new GpuInterface(*dev_it));
            }
        }
    }
}

}
}
