
#define BOOST_TEST_MAIN

#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/Types.hpp"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include <boost/any.hpp>
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

struct Paramed_object: public IMetaObject
{
    MO_BEGIN(Paramed_object);
        PARAM(int, int_value, 0);
        PARAM(float, float_value, 0);
        PARAM(double, double_value, 0);

        INPUT(int, int_input, 0);
        OUTPUT(int, int_output, 0);
    MO_END;
    void update(int value)
    {
        this->updateParam<int>("int_value", value);
    }
};

template<class T> struct TagType
{

};

namespace tag
{
    struct test_timestamp
    {
        test_timestamp& operator= (const mo::Time_t& type){data = &type; return *this;}
        const void* data = nullptr;
        static test_timestamp instance;
    };
    test_timestamp test_timestamp::instance;
    static test_timestamp& _test_timestamp = test_timestamp::instance;
}

template<class Tag> Tag indexArgs()
{
    return Tag::instance;
}

template<class Tag, class T, class ... Args> Tag indexArgs(T arg, Args... args)
{
    if(std::is_same<Tag, T>::value)
        return arg;
    return indexArgs<Tag>(args...);
}

template<class T, class ... Args>
void func(const T& data, Args... args)
{
    //auto value = indexArgs<::tag::test_timestamp, Args...>(args...);
}


MO_REGISTER_OBJECT(Paramed_object)

BOOST_AUTO_TEST_CASE(wrapped_Param)
{
    func(10, ::tag::_test_timestamp = mo::Time_t(-1 * mo::second));
    int value = 10;
    TParamPtr<int> param("Test wrapped param", &value);
    ParamTraits<int>::Storage_t data;
    BOOST_REQUIRE(param.getData(data));
    BOOST_REQUIRE_EQUAL(data, 10);
    param.updateData(5);
    BOOST_REQUIRE(param.getData(data));
    BOOST_CHECK_EQUAL(data, 5);
    param.updateData(10, mo::tag::_timestamp = mo::Time_t(1 * mo::second));
    BOOST_REQUIRE(param.getData(data));
    BOOST_CHECK_EQUAL(data, 10);
    BOOST_CHECK_EQUAL(*param.getTimestamp(), mo::Time_t(1 * mo::second));
    value = 11;
    BOOST_REQUIRE(param.getData(data));
    BOOST_CHECK_EQUAL(data, 11);
    bool update_handler_called = false;
    TSlot<void(IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags)>
        slot([&param, &update_handler_called](IParam* param_in, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags){
        update_handler_called = param_in == &param;
    });
    auto connection = param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);
    param.updateData(5);
    BOOST_REQUIRE_EQUAL(update_handler_called, true);
}


BOOST_AUTO_TEST_CASE(enum_params){
    mo::EnumParam enum_param = {{"test", 5}};

}

BOOST_AUTO_TEST_CASE(input_param){
    int value = 10;
    TParamPtr<int> param("Test wrapped param", &value);
    ITInputParam<int> input_param;
    ParamTraits<int>::Storage_t data;
    BOOST_REQUIRE(input_param.setInput(&param));
    input_param.getData(data);
    BOOST_REQUIRE_EQUAL(data, value);

    bool update_handler_called = false;
    TSlot<void(IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags)>
        slot([&update_handler_called](IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags){
        update_handler_called = true;
    });
    auto connection = input_param.registerUpdateNotifier(&slot);
    BOOST_REQUIRE(connection);
    param.updateData(5);
    BOOST_REQUIRE_EQUAL(update_handler_called, true);
}

BOOST_AUTO_TEST_CASE(access_Param)
{
    MetaObjectFactory::Instance()->RegisterTranslationUnit();

    auto obj = rcc::shared_ptr<Paramed_object>::Create();
    obj->getParam<int>("int_value");
    obj->getParam<double>("double_value");
    BOOST_REQUIRE_EQUAL(obj->getParamValue<int>("int_value"), 0);
    obj->update(10);
    BOOST_REQUIRE_EQUAL(obj->getParamValue<int>("int_value"), 10);

}

