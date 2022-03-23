#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriberPtr.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>

#include <MetaObject/thread/fiber_include.hpp>
#include <ct/reflect/print.hpp>

#include <cereal/types/vector.hpp>

#include <gtest/gtest.h>

using namespace mo;

static_assert(std::is_trivially_copy_constructible<mo::IParam>::value == false, "");
static_assert(std::is_trivially_copy_constructible<const mo::IParam>::value == false, "");
// static_assert(std::is_trivially_copy_constructible<const mo::IParam&>::value == false, "");

namespace ct
{
    template <class T, class A>
    bool operator==(const ct::TArrayView<T>& lhs, const std::vector<T, A>& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); ++i)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }
} // namespace ct

namespace
{
    template <class T>
    struct TestData
    {
        static int init(int val)
        {
            return val;
        }

        static int update(int val)
        {
            return val;
        }

        static bool check(int val, int last_update)
        {
            return val == last_update;
        }
    };

    template <class T, class A>
    struct TestData<std::vector<T, A>>
    {
        static std::vector<T, mo::TStlAllocator<T>> init(T val)
        {
            return std::vector<T, mo::TStlAllocator<T>>(val, val);
        }

        static std::vector<T, mo::TStlAllocator<T>> update(T val)
        {
            return init(val);
        }

        static bool check(const std::vector<T, A>& val, T last_update)
        {
            if (val.size() != last_update)
            {
                return false;
            }
            for (const auto& v : val)
            {
                if (v != last_update)
                {
                    return false;
                }
            }

            return true;
        }
    };

    template <class T>
    struct PublishSubscribe;

    template <class T, class U>
    struct PublishSubscribe<ct::VariadicTypedef<T, U>> : ::testing::Test
    {

        T value;
        mo::TPublisher<T> output_param;

        mo::TSubscriber<U> input_param;
        std::vector<std::pair<const IParam*, UpdateFlags>> update_flag;

        TSlot<Update_s> update_slot;
        TSlot<DataUpdate_s> data_slot;
        std::shared_ptr<Connection> connection;

        typename TSubscriber<U>::type update_val;

        IAsyncStreamPtr_t m_stream;
        bool update_called = false;

        PublishSubscribe()
        {
            output_param.setName("pub");
            input_param.setName("sub");

            m_stream = IAsyncStream::create();
            output_param.setStream(*m_stream);
            input_param.setStream(*m_stream);
            update_slot.bind(&PublishSubscribe<ct::VariadicTypedef<T, U>>::onUpdate, this);

            data_slot.bind(&PublishSubscribe<ct::VariadicTypedef<T, U>>::onDataUpdate, this);
        }

        template <class V>
        void init(V&& data)
        {
            value = TestData<T>::init(std::move(data));
        }

        void onUpdate(const IParam& param, Header, UpdateFlags fg, IAsyncStream*)
        {
            update_called = true;
            update_flag.emplace_back(&param, fg);
        }

        void onDataUpdate(const std::shared_ptr<const IDataContainer>& data,
                          const IParam& param,
                          UpdateFlags fg,
                          IAsyncStream*)
        {
            update_called = true;
            update_flag.emplace_back(&param, fg);
            auto typed = std::dynamic_pointer_cast<const TDataContainer<U>>(data);
            ASSERT_NE(typed, nullptr);
            update_val = typed->data;
        }

        void
        onTypedDataUpdate(const TDataContainerConstPtr_t<U>& data, const IParam& param, UpdateFlags fg, IAsyncStream&)
        {
            update_called = true;
            update_flag.emplace_back(&param, fg);
            update_val = data->data;
        }

        void testInitialization()
        {
            init(10);
            ASSERT_EQ(input_param.checkFlags(mo::ParamFlags::kINPUT), true);
            ASSERT_EQ(input_param.getName(), "sub");
            ASSERT_FALSE(input_param.getCurrentData());
            // ASSERT_FALSE(input_param.getCurrentData().valid());
        }

        void testGetData()
        {
            init(10);
            output_param.publish(int(this->value));
            typename TSubscriber<U>::type data;
            ASSERT_TRUE(input_param.setInput(&this->output_param));
            ASSERT_TRUE(input_param.getData(data));
            ASSERT_EQ(data, value);
        }

        void testSubscribeCallback()
        {
            init(10);
            connection = input_param.registerUpdateNotifier(update_slot);
            ASSERT_NE(connection.get(), nullptr);
            EXPECT_FALSE(update_called);
            input_param.setInput(&output_param);
            EXPECT_TRUE(update_called);
            ASSERT_EQ(update_flag.size(), 1);
            EXPECT_EQ(update_flag[0].second, ct::value(UpdateFlags::kINPUT_SET));
        }

        void testUpdateCallback()
        {
            init(10);
            ASSERT_EQ(input_param.setInput(&output_param), true);

            connection = input_param.registerUpdateNotifier(update_slot);
            ASSERT_NE(connection, nullptr);
            EXPECT_FALSE(update_called);
            output_param.publish(TestData<T>::update(5));
            EXPECT_TRUE(update_called);
            const auto count = std::count_if(
                update_flag.begin(), update_flag.end(), [](std::pair<const IParam*, UpdateFlags> val) -> bool {
                    return val.second == UpdateFlags::kINPUT_UPDATED;
                });
            EXPECT_EQ(count, 1);
        }
    };
} // namespace

TYPED_TEST_SUITE_P(PublishSubscribe);

using PubSubTestTypes = ::testing::Types<ct::VariadicTypedef<int, int>>;

// ct::VariadicTypedef<std::vector<int>, std::vector<int>>,
// ct::VariadicTypedef<std::vector<int>, ct::TArrayView<int>>

TYPED_TEST_P(PublishSubscribe, initialization)
{
    this->testInitialization();
}

TYPED_TEST_P(PublishSubscribe, getData)
{
    this->testGetData();
}

TYPED_TEST_P(PublishSubscribe, subscriberCallback)
{
    this->testSubscribeCallback();
}

TYPED_TEST_P(PublishSubscribe, updateCallback)
{
    this->testUpdateCallback();
}

REGISTER_TYPED_TEST_SUITE_P(PublishSubscribe, initialization, getData, subscriberCallback, updateCallback);

INSTANTIATE_TYPED_TEST_SUITE_P(PublishSubscribeTest, PublishSubscribe, PubSubTestTypes);

/*using PublishSubscribeWrapping = PublishSubscribe<ct::VariadicTypedef<std::vector<int>, ct::TArrayView<int>>>;

TEST_F(PublishSubscribeWrapping, wrapVecWithView)
{
    this->init(10);
    this->output_param.updateData(this->value);
    ct::TArrayView<int> view;
    ASSERT_EQ(this->input_param.setInput(&this->output_param), true);
    ASSERT_EQ(this->input_param.getTypedData(&view), true);
    ASSERT_EQ(view, this->value);

    TSubscriberPtr<ct::TArrayView<int>> wrapped_sub;
    ASSERT_EQ(wrapped_sub.setInput(&this->output_param), true);

    auto wrapped_data = wrapped_sub.getTypedData();
    ASSERT_NE(wrapped_data, nullptr);
    ASSERT_EQ(wrapped_data->data.size(), value.size());
    for (size_t i = 0; i < wrapped_data->data.size(); ++i)
    {
        ASSERT_EQ(wrapped_data->data[i], value[i]);
    }
}
*/
