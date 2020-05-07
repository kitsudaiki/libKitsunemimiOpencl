#ifndef TEST_RUN_H
#define TEST_RUN_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <libKitsunemimiCommon/test_helper/speed_test_helper.h>

namespace Kitsunemimi
{
namespace Opencl
{

class SimpleTest
        : public Kitsunemimi::SpeedTestHelper
{
public:
    SimpleTest();

    void simple_test();

    TimerSlot m_initTimeSlot;
    TimerSlot m_copyToDeviceTimeSlot;
    TimerSlot m_runTimeSlot;
    TimerSlot m_updateTimeSlot;
    TimerSlot m_copyToHostTimeSlot;
    TimerSlot m_cleanupTimeSlot;
};

}
}

#endif // TEST_RUN_H
