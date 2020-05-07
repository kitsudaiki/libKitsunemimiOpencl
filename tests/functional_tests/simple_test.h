#ifndef TEST_RUN_H
#define TEST_RUN_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{
namespace Opencl
{

class SimpleTest
        : public Kitsunemimi::CompareTestHelper
{
public:
    SimpleTest();

    void simple_test();
};

}
}

#endif // TEST_RUN_H
