#include <MetaObject/params/TMultiPublisher.hpp>
#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <ct/variadic_indexing.hpp>

#include <gtest/gtest.h>

#include <iostream>

bool printInputs(const std::tuple<const int*, const float*, const double*>& inputs)
{
    if (std::get<0>(inputs))
    {
        std::cout << "[int] " << *std::get<0>(inputs) << std::endl;
        return true;
    }
    if (std::get<1>(inputs))
    {
        std::cout << "[float] " << *std::get<1>(inputs) << std::endl;
        return true;
    }
    if (std::get<2>(inputs))
    {
        std::cout << "[double] " << *std::get<2>(inputs) << std::endl;
        return true;
    }
    std::cout << "No input set" << std::endl;
    return false;
}

void clearInputs(std::tuple<const int*, const float*, const double*>& inputs)
{
    std::get<0>(inputs) = nullptr;
    std::get<1>(inputs) = nullptr;
    std::get<2>(inputs) = nullptr;
}

namespace
{
    struct multi_type_subscriber : ::testing::Test
    {
        multi_type_subscriber()
            : m_publishers{int_out, float_out, double_out}
        {
            multi_input.setMtx(mtx);
            multi_input.setUserDataPtr(&inputs);
        }

        void checkInit()
        {
            ASSERT_EQ(multi_input.getPublisher(), nullptr);
        }

        template <class T>
        void testInput(T val)
        {
            mo::TPublisher<T>& pub = ct::get<mo::TPublisher<T>&>(m_publishers);
            ASSERT_EQ(mo::get<const T*>(inputs), static_cast<void*>(nullptr));
            ASSERT_TRUE(multi_input.setInput(&pub));

            ASSERT_NE(multi_input.getPublisher(), nullptr);

            ASSERT_EQ(multi_input.getPublisher(), &pub);
            pub.publish(val);

            ASSERT_NE(mo::get<const T*>(inputs), static_cast<void*>(nullptr));
            ASSERT_EQ(*mo::get<const T*>(inputs), 6);
            ASSERT_EQ(printInputs(inputs), true);

            pub.publish(T(5));

            ASSERT_NE(mo::get<const T*>(inputs), static_cast<void*>(nullptr));
            ASSERT_EQ(*mo::get<const T*>(inputs), 5);
            mo::Header header;
            auto data = multi_input.getData();
            ASSERT_EQ(printInputs(inputs), true);

            ASSERT_NE(data, nullptr);
            ASSERT_EQ(data->getType(), mo::TypeInfo::create<T>());
        }

        template <class T>
        void testCallbacks()
        {
            int callback_called = 0;
            auto cb = [&callback_called](
                          const mo::IDataContainerConstPtr_t&, const mo::IParam&, mo::UpdateFlags, mo::IAsyncStream&) {
                ++callback_called;
            };
            mo::TSlot<mo::DataUpdate_s> callback(std::move(cb));

            auto connection = multi_input.registerUpdateNotifier(callback);

            ASSERT_NE(connection, nullptr);
            ct::get<mo::TPublisher<T>&>(m_publishers).publish(static_cast<T>(10));
            ASSERT_EQ(callback_called, 1);
        }

        std::tuple<const int*, const float*, const double*> inputs;
        int int_val;
        mo::TPublisher<int> int_out;

        float float_val;
        mo::TPublisher<float> float_out;

        double double_val;
        mo::TPublisher<double> double_out;

        std::tuple<mo::TPublisher<int>&, mo::TPublisher<float>&, mo::TPublisher<double>&> m_publishers;

        mo::TMultiSubscriber<int, float, double> multi_input;

        mo::Mutex_t mtx;

        mo::TMultiOutput<int, float, double> multi_output;
    };
} // namespace

TEST_F(multi_type_subscriber, initialization)
{
    checkInit();
    ASSERT_NE(printInputs(inputs), true);
}

TEST_F(multi_type_subscriber, int_subscribe)
{
    auto stream = mo::IAsyncStream::create();
    testInput<int>(6);
    testCallbacks<int>();
}

TEST_F(multi_type_subscriber, float_subscribe)
{
    auto stream = mo::IAsyncStream::create();
    testInput<float>(6);
    testCallbacks<float>();
}

TEST_F(multi_type_subscriber, double_subscribe)
{
    auto stream = mo::IAsyncStream::create();
    testInput<double>(6);
    testCallbacks<double>();
}
