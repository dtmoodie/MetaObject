
#include "MetaObject/core.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/types/file_types.hpp"
#include "MetaObject/runtime_reflection.hpp"

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <MetaObject/thread/fiber_include.hpp>
#include <boost/any.hpp>


#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <iostream>

using namespace mo;

template <class T>
struct WrappedParam
{
    T value = 10;
    TParamPtr<T> param;
    WrappedParam()
        : param("test", &value)
    {
    }
};

struct TestReadVisitor : public LoadCache
{
    VisitorTraits traits() const
    {
        return {true, true};
    }

    ILoadVisitor& operator()(ILoadContainerTraits*, const std::string&)
    {
        return *this;
    }

    virtual ILoadVisitor& operator()(bool*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }

    virtual ILoadVisitor& operator()(char*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(int8_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(uint8_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(int16_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(uint16_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(int32_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(uint32_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(int64_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(uint64_t*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
#ifdef ENVIRONMENT64
    virtual ILoadVisitor& operator()(long long*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }

    virtual ILoadVisitor& operator()(unsigned long long*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
#else
    virtual ILoadVisitor& operator()(long int*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }

    virtual ILoadVisitor& operator()(unsigned long int*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
#endif
    virtual ILoadVisitor& operator()(float*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(double*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    virtual ILoadVisitor& operator()(void*, const std::string& = "", const size_t = 1)
    {
        ++count;
        return *this;
    }
    std::string getCurrentElementName() const
    {
        return "";
    }
    int count = 0;
};

BOOST_FIXTURE_TEST_CASE(set_value, WrappedParam<int>)
{
    param.updateData(100);
    BOOST_REQUIRE_EQUAL(value, 100);
}

BOOST_FIXTURE_TEST_CASE(param_timestamping, WrappedParam<int>)
{
    param.updateData(100, tag::_timestamp = 1 * ms);
    BOOST_REQUIRE(param.getTimestamp());
    BOOST_REQUIRE_EQUAL(param.getTimestamp()->time_since_epoch(), 1 * ms);
    BOOST_REQUIRE_EQUAL(param.getFrameNumber(), 0);

    param.updateData(101, tag::_timestamp = 2 * ms);
    BOOST_REQUIRE(param.getTimestamp());
    BOOST_REQUIRE_EQUAL(param.getTimestamp()->time_since_epoch(), 2 * ms);
    BOOST_REQUIRE_EQUAL(param.getFrameNumber(), 1);
}

BOOST_FIXTURE_TEST_CASE(read_from_param, WrappedParam<int>)
{
    param.updateData(100);
    BOOST_REQUIRE(param.getData());

    int val;
    BOOST_REQUIRE(param.getTypedData(&val));
    BOOST_REQUIRE_EQUAL(val, 100);

    auto container = param.getTypedData<int>(Header());
    BOOST_REQUIRE(container);
    BOOST_REQUIRE_EQUAL(container->data, 100);
}

BOOST_FIXTURE_TEST_CASE(read_async_param, WrappedParam<int>)
{
    param.updateData(100, Header(mo::Time(mo::ms * 33)));
    BOOST_REQUIRE(param.getData());
}

BOOST_FIXTURE_TEST_CASE(read_sync_param, WrappedParam<int>)
{
    param.updateData(100, Header(mo::Time(mo::ms * 33)));
    BOOST_REQUIRE_NE(param.getData(Header(mo::Time(mo::ms * 33))).get(), (void*)nullptr);
    auto container = param.getData(Header(mo::Time(mo::ms * 34)));
    BOOST_REQUIRE(container == nullptr);

    param.updateData(100);
    int data;
    BOOST_REQUIRE(param.getTypedData(&data));
    BOOST_REQUIRE(!param.getTypedData(&data, Header(mo::Time(mo::ms * 33))));

    param.updateData(10, mo::tag::_timestamp = mo::Time(1 * mo::second));
    BOOST_REQUIRE(param.getTypedData(&data));
    BOOST_REQUIRE(!param.getTypedData(&data, Header(mo::Time(mo::ms * 33))));
    BOOST_REQUIRE(param.getTypedData(&data, Header(mo::Time(mo::second * 1))));
}

BOOST_FIXTURE_TEST_CASE(read_visit_param, WrappedParam<int>)
{
    param.updateData(100);
    TestReadVisitor visitor;
    BOOST_REQUIRE_EQUAL(visitor.count, 0);
    param.load(visitor);
    // visit header.frame_number, header.timestamp, and param.data
    BOOST_REQUIRE_EQUAL(visitor.count, 3);
}

BOOST_FIXTURE_TEST_CASE(param_update, WrappedParam<int>)
{
    bool update_handler_called = false;
    TSlot<void(IParam*, Header, UpdateFlags)> slot(
        [this, &update_handler_called](IParam* param_in, Header, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            BOOST_REQUIRE(fg == UpdateFlags::ValueUpdated_e);
        });

    auto connection = param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);

    param.updateData(100);

    BOOST_REQUIRE(update_handler_called);
}

BOOST_FIXTURE_TEST_CASE(param_data_update, WrappedParam<int>)
{
    bool update_handler_called = false;
    TSlot<void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags)> slot(
        [this, &update_handler_called](const std::shared_ptr<IDataContainer>&, IParam* param_in, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            BOOST_REQUIRE(fg == UpdateFlags::ValueUpdated_e);
        });

    auto connection = param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);

    param.updateData(100);

    BOOST_REQUIRE(update_handler_called);
}

BOOST_FIXTURE_TEST_CASE(param_typed_data_update, WrappedParam<int>)
{
    bool update_handler_called = false;
    TSlot<void(mo::TParam<int>::TContainerPtr_t, IParam*, UpdateFlags)> slot(
        [this, &update_handler_called](TParam<int>::TContainerPtr_t, IParam* param_in, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            BOOST_REQUIRE(fg == UpdateFlags::ValueUpdated_e);
        });

    auto connection = param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);

    param.updateData(100);
    BOOST_REQUIRE(update_handler_called);
}
