
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Params//ParamMacros.hpp"
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Params/Types.hpp"
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
        this->UpdateParam<int>("int_value", value);
    }
};

struct ParamUpdateToken
{
    ParamUpdateToken(mo::IParam& param):
        _param(param)
    {
    }
    ~ParamUpdateToken()
    {
        //if(_timestamp_changed)
            
    }
    ParamUpdateToken& operator()(mo::Context* ctx)
    {
        _ctx = ctx;
        return *this;
    }
    ParamUpdateToken& operator()(long long fn)
    {
        _frame_number = fn;
        return *this;
    }
    ParamUpdateToken& operator()(mo::time_t time)
    {
        _timestamp = time;
        _timestamp_changed = true;
        return *this;
    }
    long long _frame_number = -1;
    mo::time_t _timestamp;
    mo::Context* _ctx = nullptr;
    bool _timestamp_changed;
    mo::IParam& _param;
};

template<class T> struct TagType
{

};

namespace tag
{
    struct test_timestamp
    {
        test_timestamp& operator= (const mo::time_t& type){data = &type; return *this;}
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
    func(10, ::tag::_test_timestamp = mo::time_t(-1 * mo::second));
	int value = 10;
	TParamPtr<int> param("Test wrapped param", &value);

	BOOST_CHECK_EQUAL(param.GetData(), 10);
	param.UpdateData(5);
	BOOST_CHECK_EQUAL(param.GetData(), 5);
    param.UpdateData(10, mo::tag::_timestamp = mo::time_t(1 * mo::second));
    BOOST_CHECK_EQUAL(param.GetData(), 10);
    BOOST_CHECK_EQUAL(*param.GetTimestamp(), mo::time_t(1 * mo::second));
	value = 11;
	BOOST_CHECK_EQUAL(param.GetData(), 11);
	bool update_handler_called = false;
	TSlot<void(Context*, IParam*)> slot([&param, &update_handler_called](Context* ctx, IParam* param_in)
	{
		update_handler_called = param_in == &param;
	});
	param.RegisterUpdateNotifier(&slot);
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);
}


BOOST_AUTO_TEST_CASE(enum_params)
{
    mo::EnumParam enum_param = {{"test", 5}};
    
}

BOOST_AUTO_TEST_CASE(input_Param)
{
	int value = 10;
	TParamPtr<int> param("Test wrapped param", &value);
    ITInputParam<int> input_param;
	BOOST_REQUIRE(input_param.setInput(&param));
	BOOST_REQUIRE_EQUAL(input_param.GetData(), value);
	
	bool update_handler_called = false;
	TSlot<void(Context*, IParam*)> slot(
		[&update_handler_called](Context*, IParam*)
	{
		update_handler_called = true;
	});

	BOOST_REQUIRE(input_param.RegisterUpdateNotifier(&slot));
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);
}

BOOST_AUTO_TEST_CASE(access_Param)
{
    MetaObjectFactory::Instance()->RegisterTranslationUnit();

    auto obj = rcc::shared_ptr<Paramed_object>::Create();
    obj->GetParam<int>("int_value");
    obj->GetParam<double>("double_value");
    BOOST_REQUIRE_EQUAL(obj->GetParamValue<int>("int_value"), 0);
    obj->update(10);
    BOOST_REQUIRE_EQUAL(obj->GetParamValue<int>("int_value"), 10);

}

