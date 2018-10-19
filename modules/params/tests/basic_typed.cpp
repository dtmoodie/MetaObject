
#define BOOST_TEST_MAIN

#include "MetaObject/core.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/types/file_types.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include <boost/any.hpp>
#include <boost/thread/recursive_mutex.hpp>
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(wrapped_Param)
{
    int value = 10;
    TParamPtr<int> param("Test wrapped param", &value);
    param.updateData(value);
    auto container = param.getTypedData<int>();
    BOOST_REQUIRE(container);
    BOOST_REQUIRE_EQUAL(container->data, 10);
    param.updateData(5);
    container = param.getTypedData<int>();
    BOOST_REQUIRE(container);
    BOOST_CHECK_EQUAL(container->data, 5);
    param.updateData(10, mo::tag::_timestamp = mo::Time(1 * mo::second));
    int data;
    BOOST_REQUIRE(param.getTypedData(&data));
    BOOST_CHECK_EQUAL(data, 10);
    BOOST_CHECK_EQUAL(*param.getTimestamp(), mo::Time(1 * mo::second));
    param.updateData(11);
    BOOST_CHECK_EQUAL(value, 11);
    BOOST_REQUIRE(param.getTypedData(&data));
    BOOST_CHECK_EQUAL(data, 11);
    bool update_handler_called = false;

    TSlot<void(IParam*, Header, UpdateFlags)> slot([&param, &update_handler_called](
        IParam* param_in, Header, UpdateFlags) { update_handler_called = param_in == &param; });
    auto connection = param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);
    param.updateData(5);
    BOOST_REQUIRE_EQUAL(update_handler_called, true);
}

BOOST_AUTO_TEST_CASE(enum_params)
{
    mo::EnumParam enum_param = {{"test", 5}};
}

BOOST_AUTO_TEST_CASE(input_param)
{
    int value = 10;
    TParamPtr<int> param("Test wrapped param", &value);
    param.updateData(value);
    ITInputParam<int> input_param;
    int data;
    BOOST_REQUIRE(input_param.setInput(&param));
    BOOST_REQUIRE(input_param.getTypedData(&data));
    BOOST_REQUIRE_EQUAL(data, value);

    bool update_handler_called = false;
    TSlot<void(IParam*, Header, UpdateFlags)> slot(
        [&update_handler_called](IParam*, Header, UpdateFlags) { update_handler_called = true; });
    auto connection = input_param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);
    param.updateData(5);
    BOOST_REQUIRE_EQUAL(update_handler_called, true);
}
