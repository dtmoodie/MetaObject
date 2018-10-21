#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>

using namespace mo;

template <class T>
struct Fixture
{
    T value = 10;
    TParamPtr<T> param;

    ITInputParam<int> input_param;

    bool update_called = false;
    UpdateFlags update_flag;
    Fixture()
        : param("test", &value)
    {
        update_slot =
            std::bind(&Fixture<T>::onUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        data_slot = std::bind(
            &Fixture<T>::onDataUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        typed_slot = std::bind(
            &Fixture<T>::onTypedDataUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    }

    void onUpdate(IParam*, Header, UpdateFlags fg)
    {
        update_called = true;
        update_flag = fg;
    }

    void onDataUpdate(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags fg)
    {
        update_called = true;
        update_flag = fg;
    }

    void onTypedDataUpdate(mo::TParam<int>::TContainerPtr_t, IParam*, UpdateFlags fg)
    {
        update_called = true;
        update_flag = fg;
    }

    TSlot<void(IParam*, Header, UpdateFlags)> update_slot;
    TSlot<void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags)> data_slot;
    TSlot<void(mo::TParam<int>::TContainerPtr_t, IParam*, UpdateFlags)> typed_slot;
    std::shared_ptr<Connection> connection;
};

BOOST_FIXTURE_TEST_CASE(input_param_get_data, Fixture<int>)
{
    param.updateData(value);
    int data;
    BOOST_REQUIRE(input_param.setInput(&param));
    BOOST_REQUIRE(input_param.getTypedData(&data));
    BOOST_REQUIRE_EQUAL(data, value);
}

BOOST_FIXTURE_TEST_CASE(input_param_subscription_callback, Fixture<int>)
{
    connection = input_param.registerUpdateNotifier(&update_slot);
    BOOST_REQUIRE(connection);

    input_param.setInput(&param);
    BOOST_REQUIRE(update_called);
    BOOST_REQUIRE(update_flag == UpdateFlags::InputSet_e);
}

BOOST_FIXTURE_TEST_CASE(input_param_update_callback, Fixture<int>)
{
    BOOST_REQUIRE(input_param.setInput(&param));

    connection = input_param.registerUpdateNotifier(&update_slot);
    BOOST_REQUIRE(connection);
    BOOST_REQUIRE(update_called == false);
    param.updateData(5);
    BOOST_REQUIRE(update_called == true);
    BOOST_REQUIRE(update_flag == UpdateFlags::InputUpdated_e);
}
