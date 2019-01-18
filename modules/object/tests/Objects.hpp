#pragma once

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/object.hpp>

namespace test
{
    using namespace mo;

    struct SerializableObject : public MetaObject
    {
        ~SerializableObject() override;
        MO_BEGIN(SerializableObject)
           PARAM(int, test, 5)
           PARAM(int, test2, 6)
        MO_END
    };
}
