#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>

#include <MetaObject/thread/fiber_include.hpp>
#include <ct/reflect/print.hpp>

#include <cereal/types/vector.hpp>

#include <gtest/gtest.h>

using namespace mo;

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
}

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
        static std::vector<T, mo::TVectorAllocator<T>> init(T val)
        {
            return std::vector<T, mo::TVectorAllocator<T>>(val, val);
        }

        static std::vector<T, mo::TVectorAllocator<T>> update(T val)
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
        using type = typename TParamPtr<T>::type;

        type value;
        TParamPtr<T> output_param;

        ITInputParam<U> input_param;
        std::vector<std::pair<IParam*, UpdateFlags>> update_flag;

        TSlot<void(IParam*, Header, UpdateFlags)> update_slot;
        TSlot<void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags)> data_slot;
        TSlot<void(typename mo::TParam<U>::TContainerPtr_t, IParam*, UpdateFlags)> typed_slot;
        std::shared_ptr<Connection> connection;

        typename ITInputParam<U>::type update_val;
        bool update_called = false;

        PublishSubscribe()
            : output_param("pub", &value)
            , input_param("sub")
        {
            update_slot = std::bind(&PublishSubscribe<ct::VariadicTypedef<T, U>>::onUpdate,
                                    this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3);

            data_slot = std::bind(&PublishSubscribe<ct::VariadicTypedef<T, U>>::onDataUpdate,
                                  this,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3);

            typed_slot = std::bind(&PublishSubscribe<ct::VariadicTypedef<T, U>>::onTypedDataUpdate,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2,
                                   std::placeholders::_3);
        }

        template <class V>
        void init(V&& data)
        {
            value = TestData<type>::init(std::move(data));
        }

        void onUpdate(IParam* param, Header, UpdateFlags fg)
        {
            update_called = true;
            update_flag.emplace_back(param, fg);
        }

        void onDataUpdate(const std::shared_ptr<IDataContainer>& data, IParam* param, UpdateFlags fg)
        {
            update_called = true;
            update_flag.emplace_back(param, fg);
            auto typed = std::dynamic_pointer_cast<TDataContainer<U>>(data);
            ASSERT_NE(typed, nullptr);
            update_val = typed->data;
        }

        void onTypedDataUpdate(typename mo::TParam<U>::TContainerPtr_t data, IParam* param, UpdateFlags fg)
        {
            update_called = true;
            update_flag.emplace_back(param, fg);
            update_val = data->data;
        }

        void testInitialization()
        {
            init(10);
            ASSERT_EQ(input_param.checkFlags(mo::ParamFlags::kINPUT), true);
            ASSERT_EQ(input_param.getName(), "sub");
            ASSERT_FALSE(input_param.getInputFrameNumber().valid());
        }

        void testGetData()
        {
            init(10);
            output_param.updateData(this->value);
            typename ITInputParam<U>::type data;
            ASSERT_TRUE(input_param.setInput(&this->output_param));
            ASSERT_TRUE(input_param.getTypedData(&data));
            ASSERT_EQ(data, value);
        }

        void testSubscribeCallback()
        {
            init(10);
            connection = input_param.registerUpdateNotifier(&update_slot);
            ASSERT_NE(connection.get(), nullptr);
            EXPECT_FALSE(update_called);
            input_param.setInput(&output_param);
            EXPECT_TRUE(update_called);
            ASSERT_EQ(update_flag.size(), 3);
            EXPECT_EQ(update_flag[0].second, ct::value(UpdateFlags::kINPUT_SET));
            EXPECT_EQ(update_flag[1].second, ct::value(UpdateFlags::kINPUT_UPDATED));
            EXPECT_EQ(update_flag[2].second, ct::value(UpdateFlags::kVALUE_UPDATED));

        }

        void testUpdateCallback()
        {
            init(10);
            ASSERT_EQ(input_param.setInput(&output_param), true);

            connection = input_param.registerUpdateNotifier(&update_slot);
            ASSERT_NE(connection, nullptr);
            EXPECT_FALSE(update_called);
            output_param.updateData(TestData<T>::update(5));
            EXPECT_TRUE(update_called);
            const auto count = std::count_if(update_flag.begin(), update_flag.end(), [](std::pair<IParam*, UpdateFlags> val)->bool{return val.second == UpdateFlags::kINPUT_UPDATED;});
            EXPECT_EQ(count, 1);
        }

        void testTypedCallback()
        {
            init(10);
            ASSERT_EQ(input_param.setInput(&output_param), true);

            connection = input_param.registerUpdateNotifier(&typed_slot);
            ASSERT_NE(connection.get(), nullptr);
            ASSERT_FALSE(update_called);
            output_param.updateData(TestData<T>::update(5));
            ASSERT_TRUE(update_called);
            const auto count = std::count_if(update_flag.begin(), update_flag.end(), [](std::pair<IParam*, UpdateFlags> val)->bool{return val.second == UpdateFlags::kINPUT_UPDATED;});
            ASSERT_EQ(count, 1);
            ASSERT_EQ(update_val, TestData<T>::update(5));
        }
    };
}

TYPED_TEST_SUITE_P(PublishSubscribe);

using PubSubTestTypes = ::testing::Types<ct::VariadicTypedef<int, int>>;

//ct::VariadicTypedef<std::vector<int>, std::vector<int>>,
//ct::VariadicTypedef<std::vector<int>, ct::TArrayView<int>>

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

TYPED_TEST_P(PublishSubscribe, typedUpdateCallback)
{
    this->testTypedCallback();
}

REGISTER_TYPED_TEST_SUITE_P(PublishSubscribe, initialization, getData, subscriberCallback, updateCallback, typedUpdateCallback);

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

    TInputParamPtr<ct::TArrayView<int>> wrapped_sub;
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
