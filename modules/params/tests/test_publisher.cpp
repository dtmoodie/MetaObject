
#include "MetaObject/core.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TPublisher.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
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
struct publisher_ : public ::testing::Test
{
    T value = 10;
    TPublisher<T> param;
    IAsyncStreamPtr_t m_stream;
    publisher_()
    {
        m_stream = IAsyncStream::create();
        param.setStream(*m_stream);
        param.setName("test");
    }
};

using publisher = publisher_<int>;

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

TEST_F(publisher, set_value)
{
    param.publish(100);
}

TEST_F(publisher, publish_timestamping)
{
    param.publish(100, tags::timestamp = 1 * ms);
    auto data = param.getData();
    ASSERT_NE(data->getHeader().timestamp, boost::none);
    ASSERT_EQ(data->getHeader().timestamp->time_since_epoch(), 1 * ms);
    ASSERT_EQ(data->getHeader().frame_number, 0);

    param.publish(101, tags::timestamp = 2 * ms);
    data = param.getData();
    ASSERT_NE(data->getHeader().timestamp, boost::none);
    ASSERT_EQ(data->getHeader().timestamp->time_since_epoch(), 2 * ms);
    ASSERT_EQ(data->getHeader().frame_number, 1);
}

TEST_F(publisher, get_data)
{
    param.publish(100);
    auto data = param.getData();
    ASSERT_NE(data, nullptr);
    auto typed = std::dynamic_pointer_cast<const mo::TDataContainer<int>>(data);
    ASSERT_NE(typed, nullptr);

    ASSERT_EQ(typed->data, 100);
}

TEST_F(publisher, get_data_timestamped)
{
    Header header(mo::Time(mo::ms * 33));
    param.publish(100, header);
    auto data = param.getData(&header);
    ASSERT_NE(data, nullptr) << "We expect to be able to get data based on this header, since it was used to publish";

    header = Header(mo::Time(mo::ms * 34));
    data = param.getData(&header);
    ASSERT_EQ(data, nullptr) << "We've not published data at this timestamp, we should not be getting data";

    param.publish(100);
    data = param.getData();
    ASSERT_NE(data, nullptr)
        << "We've published data with an empty header, so we should be able to fetch it without providing a header";
    data = param.getData(&header);
    ASSERT_EQ(data, nullptr) << "We've published data without a header, so we should not expect to get data when "
                                "providing a specific timestamp";
}

/*TEST_F(publisher, visit_param)
{
    param.publish(100);
    TestReadVisitor visitor;
    ASSERT_EQ(visitor.count, 0);
    param.load(visitor);
    ASSERT_EQ(visitor.count, 4);
}*/

TEST_F(publisher, publish_callback)
{
    bool update_callback_invoked = false;

    auto cb = [this, &update_callback_invoked](const IParam& param_in, Header, UpdateFlags fg, IAsyncStream* stream) {
        update_callback_invoked = (&param_in == &param);
        ASSERT_EQ(fg, ct::value(UpdateFlags::kVALUE_UPDATED));
    };

    TSlot<Update_s> slot(std::move(cb));

    auto connection = param.registerUpdateNotifier(slot);
    ASSERT_NE(connection, nullptr);

    param.publish(100);

    ASSERT_EQ(update_callback_invoked, true);
}

TEST_F(publisher, publish_callback_data)
{
    bool update_callback_invoked = false;
    auto callback = [this, &update_callback_invoked](
                        const IDataContainerConstPtr_t&, const IParam& param_in, UpdateFlags fg, IAsyncStream* stream) {
        update_callback_invoked = (&param_in == &param);
        ASSERT_EQ(fg, ct::value(UpdateFlags::kVALUE_UPDATED));
    };

    TSlot<DataUpdate_s> slot(std::move(callback));

    auto connection = param.registerUpdateNotifier(slot);
    ASSERT_NE(connection, nullptr);

    param.publish(100);

    ASSERT_EQ(update_callback_invoked, true);
}
