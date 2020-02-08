#include <MetaObject/params/TMultiInput-inl.hpp>
#include <MetaObject/params/TMultiOutput.hpp>
#include <MetaObject/params/TParamOutput.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/thread/Mutex.hpp>

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
        std::cout << "[float] " << *std::get<0>(inputs) << std::endl;
        return true;
    }
    if (std::get<2>(inputs))
    {
        std::cout << "[double] " << *std::get<0>(inputs) << std::endl;
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
    struct Fixture : ::testing::Test
    {
        Fixture()
        {
            multi_input.setMtx(&mtx);
            multi_input.setUserDataPtr(&inputs);
        }

        void checkInit()
        {
            ASSERT_EQ(multi_input.getInputParam(), nullptr);
        }

        void testInput(int val)
        {

            ASSERT_EQ(mo::get<const int*>(inputs), static_cast<void*>(nullptr));
            ASSERT_EQ(multi_input.setInput(&int_out), true);

            ASSERT_NE(multi_input.getInputParam(), nullptr);

            ASSERT_EQ(multi_input.getInputParam(), &int_out);
            int_out.updateData(val);

            ASSERT_NE(mo::get<const int*>(inputs), static_cast<void*>(nullptr));
            ASSERT_EQ(*mo::get<const int*>(inputs), 6);
            ASSERT_EQ(printInputs(inputs), true);

            int_out.updateData(5);

            ASSERT_NE(mo::get<const int*>(inputs), static_cast<void*>(nullptr));
            ASSERT_EQ(*mo::get<const int*>(inputs), 5);

            auto data = multi_input.getData(mo::Header());
            ASSERT_EQ(printInputs(inputs), true);

            ASSERT_NE(data, nullptr);
            ASSERT_EQ(data->getType(), mo::TypeInfo(typeid(int)));
        }

        void testCallbacks()
        {
            int callback_called = 0;
            mo::TParam<int>::TUpdateSlot_t int_callback([&callback_called](mo::TParam<int>::TContainerPtr_t,
                                                                           mo::IParam*,
                                                                           mo::UpdateFlags) { ++callback_called; });
            auto connection = multi_input.registerUpdateNotifier(&int_callback);

            ASSERT_NE(connection, nullptr);
            int_out.updateData(10);
            ASSERT_EQ(callback_called, 1);
        }

        std::tuple<const int*, const float*, const double*> inputs;
        int int_val;
        mo::TParamOutput<int> int_out;

        float float_val;
        mo::TParamOutput<float> float_out;

        double double_val;
        mo::TParamOutput<double> double_out;

        mo::TMultiInput<int, float, double> multi_input;

        mo::Mutex_t mtx;

        mo::TMultiOutput<int, float, double> multi_output;
    };
} // namespace
TEST_F(Fixture, init)
{
    checkInit();
    ASSERT_NE(printInputs(inputs), true);
}

TEST_F(Fixture, int_subscribe)
{
    testInput(6);
    testCallbacks();
}
