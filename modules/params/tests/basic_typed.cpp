
#include "MetaObject/core.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/runtime_reflection.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/types/file_types.hpp"
#include <MetaObject/thread/Mutex.hpp>

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"

#include <boost/any.hpp>

#include "gtest/gtest.h"

#include <iostream>
#include <mutex>

using namespace mo;

template <class T>
struct WrappedParam_ : public ::testing::Test
{
    T value = 10;
    TParamPtr<T> param;
    WrappedParam_()
        : param("test", &value)
    {
    }
};

using WrappedParam = WrappedParam_<int>;

struct TestReadVisitor : public LoadCache
{
    VisitorTraits traits() const
    {
        return {true, true};
    }

    ILoadVisitor& operator()(IContainerTraits*, void*, const std::string&, size_t)
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
#ifndef _MSC_VER
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
#endif
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
    size_t getCurrentContainerSize() const
    {
        return 0;
    }
    int count = 0;
};

TEST_F(WrappedParam, set_value)
{
    param.updateData(100);
    ASSERT_EQ(value, 100);
}

TEST_F(WrappedParam, param_timestamping)
{
    param.updateData(100, timestamp = 1 * ms);
    ASSERT_NE(param.getTimestamp(), boost::none);
    ASSERT_EQ(param.getTimestamp()->time_since_epoch(), 1 * ms);
    ASSERT_EQ(param.getFrameNumber(), 0);

    param.updateData(101, timestamp = 2 * ms);
    ASSERT_NE(param.getTimestamp(), boost::none);
    ASSERT_EQ(param.getTimestamp()->time_since_epoch(), 2 * ms);
    ASSERT_EQ(param.getFrameNumber(), 1);
}

TEST_F(WrappedParam, read_from_param)
{
    param.updateData(100);
    ASSERT_NE(param.getData(), nullptr);

    int val;
    ASSERT_EQ(param.getTypedData(&val), true);
    ASSERT_EQ(val, 100);

    auto container = param.template getTypedData<int>(Header());
    ASSERT_NE(container, nullptr);
    ASSERT_EQ(container->data, 100);
}

TEST_F(WrappedParam, read_async_param)
{
    param.updateData(100, Header(mo::Time(mo::ms * 33)));
    ASSERT_NE(param.getData(), nullptr);
}

TEST_F(WrappedParam, read_sync_param)
{
    param.updateData(100, Header(mo::Time(mo::ms * 33)));
    ASSERT_NE(param.getData(Header(mo::Time(mo::ms * 33))).get(), (void*)nullptr);
    auto container = param.getData(Header(mo::Time(mo::ms * 34)));
    ASSERT_EQ(container, nullptr);

    param.updateData(100);
    int data;
    ASSERT_EQ(param.getTypedData(&data), true);
    ASSERT_EQ(param.getTypedData(&data, Header(mo::Time(mo::ms * 33))), false);

    param.updateData(10, mo::timestamp = 1 * mo::second);
    ASSERT_EQ(param.getTypedData(&data), true);
    ASSERT_EQ(param.getTypedData(&data, Header(mo::Time(mo::ms * 33))), false);
    ASSERT_EQ(param.getTypedData(&data, Header(mo::Time(mo::second * 1))), true);
}

TEST_F(WrappedParam, read_visit_param)
{
    param.updateData(100);
    TestReadVisitor visitor;
    ASSERT_EQ(visitor.count, 0);
    param.load(visitor);
    // visit header.frame_number, header.timestamp, and param.data
    ASSERT_EQ(visitor.count, 3);
}

TEST_F(WrappedParam, param_update)
{
    bool update_handler_called = false;
    TSlot<void(IParam*, Header, UpdateFlags)> slot(
        [this, &update_handler_called](IParam* param_in, Header, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            ASSERT_EQ(fg, ct::value(UpdateFlags::kVALUE_UPDATED));
        });

    auto connection = param.registerUpdateNotifier(&slot);
    ASSERT_NE(connection, nullptr);

    param.updateData(100);

    ASSERT_EQ(update_handler_called, true);
}

TEST_F(WrappedParam, param_data_update)
{
    bool update_handler_called = false;
    TSlot<void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags)> slot(
        [this, &update_handler_called](const std::shared_ptr<IDataContainer>&, IParam* param_in, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            ASSERT_EQ(fg, ct::value(UpdateFlags::kVALUE_UPDATED));
        });

    auto connection = param.registerUpdateNotifier(&slot);
    ASSERT_NE(connection, nullptr);

    param.updateData(100);

    ASSERT_EQ(update_handler_called, true);
}

TEST_F(WrappedParam, param_typed_data_update)
{
    bool update_handler_called = false;
    TSlot<void(mo::TParam<int>::TContainerPtr_t, IParam*, UpdateFlags)> slot(
        [this, &update_handler_called](TParam<int>::TContainerPtr_t, IParam* param_in, UpdateFlags fg) {
            update_handler_called = param_in == &param;
            ASSERT_EQ(fg, ct::value(UpdateFlags::kVALUE_UPDATED));
        });

    auto connection = param.registerUpdateNotifier(&slot);
    ASSERT_NE(connection, nullptr);

    param.updateData(100);
    ASSERT_EQ(update_handler_called, true);
}
