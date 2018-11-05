#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>

using namespace mo;

namespace
{

    template <class T>
    struct Fixture
    {
        T value = 10;
        TParamPtr<T> param;

        ITInputParam<int> input_param;

        bool update_called = false;
        std::vector<UpdateFlags> update_flag;
        T update_val;

        TSlot<void(IParam*, Header, UpdateFlags)> update_slot;
        TSlot<void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags)> data_slot;
        TSlot<void(mo::TParam<int>::TContainerPtr_t, IParam*, UpdateFlags)> typed_slot;
        std::shared_ptr<Connection> connection;

        Fixture()
            : param("pub", &value)
            , input_param("sub")
        {
            update_slot = std::bind(
                &Fixture<T>::onUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            data_slot = std::bind(
                &Fixture<T>::onDataUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            typed_slot = std::bind(&Fixture<T>::onTypedDataUpdate,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2,
                                   std::placeholders::_3);
        }

        void onUpdate(IParam*, Header, UpdateFlags fg)
        {
            update_called = true;
            update_flag.push_back(fg);
        }

        void onDataUpdate(const std::shared_ptr<IDataContainer>& data, IParam*, UpdateFlags fg)
        {
            update_called = true;
            update_flag.push_back(fg);
            auto typed = std::dynamic_pointer_cast<TDataContainer<T>>(data);
            BOOST_REQUIRE(typed);
            update_val = typed->data;
        }

        void onTypedDataUpdate(mo::TParam<int>::TContainerPtr_t data, IParam*, UpdateFlags fg)
        {
            update_called = true;
            update_flag.push_back(fg);
            update_val = data->data;
        }
    };
}

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
    BOOST_REQUIRE(update_flag.back() == UpdateFlags::InputSet_e);
    BOOST_REQUIRE(update_flag.size() == 1);
}

BOOST_FIXTURE_TEST_CASE(input_param_update_callback, Fixture<int>)
{
    BOOST_REQUIRE(input_param.setInput(&param));

    connection = input_param.registerUpdateNotifier(&update_slot);
    BOOST_REQUIRE(connection);
    BOOST_REQUIRE(update_called == false);
    param.updateData(5);
    BOOST_REQUIRE(update_called == true);
    BOOST_REQUIRE_EQUAL((std::count(update_flag.begin(), update_flag.end(), UpdateFlags::InputUpdated_e)), 1);
}

BOOST_FIXTURE_TEST_CASE(input_param_data_callback, Fixture<int>)
{
    BOOST_REQUIRE(input_param.setInput(&param));

    connection = input_param.registerUpdateNotifier(&data_slot);
    BOOST_REQUIRE(connection);
    BOOST_REQUIRE(update_called == false);
    param.updateData(5);
    BOOST_REQUIRE(update_called == true);
    BOOST_REQUIRE_EQUAL((std::count(update_flag.begin(), update_flag.end(), UpdateFlags::InputUpdated_e)), 1);
    BOOST_REQUIRE(update_val == 5);
}

BOOST_FIXTURE_TEST_CASE(input_param_typed_callback, Fixture<int>)
{
    BOOST_REQUIRE(input_param.setInput(&param));

    connection = input_param.registerUpdateNotifier(&typed_slot);
    BOOST_REQUIRE(connection);
    BOOST_REQUIRE(update_called == false);
    param.updateData(5);
    BOOST_REQUIRE(update_called == true);
    BOOST_REQUIRE(std::count(update_flag.begin(), update_flag.end(), UpdateFlags::InputUpdated_e) == 1);
    BOOST_REQUIRE(update_val == 5);
}
