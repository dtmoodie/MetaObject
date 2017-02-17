
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

struct parametered_object: public IMetaObject
{
    MO_BEGIN(parametered_object);
        PARAM(int, int_value, 0);
        PARAM(float, float_value, 0);
        PARAM(double, double_value, 0);

        INPUT(int, int_input, 0);
        OUTPUT(int, int_output, 0);
    MO_END;
    void update(int value)
    {
        this->UpdateParameter<int>("int_value", value);
    }
};

struct ParameterUpdateToken
{
    ParameterUpdateToken(mo::IParameter& param):
        _param(param)
    {
    }
    ~ParameterUpdateToken()
    {
        if(_timestamp_changed)
            
    }
    ParameterUpdateToken& operator()(mo::Context* ctx)
    {
        _ctx = ctx;
        return *this;
    }
    ParameterUpdateToken& operator()(long long fn)
    {
        _frame_number = fn;
        return *this;
    }
    ParameterUpdateToken& operator()(mo::time_t time)
    {
        _timestamp = time;
        _timestamp_changed = true;
        return *this;
    }
    long long _frame_number = -1;
    mo::time_t _timestamp;
    mo::Context* _ctx = nullptr;
    bool _timestamp_changed;
    mo::IParameter& _param;
};

MO_REGISTER_OBJECT(parametered_object)

BOOST_AUTO_TEST_CASE(wrapped_parameter)
{
	int value = 10;
	TypedParameterPtr<int> param("Test wrapped param", &value);

	BOOST_CHECK_EQUAL(param.GetData(), 10);
	param.UpdateData(5);
	BOOST_CHECK_EQUAL(param.GetData(), 5);
	value = 11;
	BOOST_CHECK_EQUAL(param.GetData(), 11);
	bool update_handler_called = false;
	TypedSlot<void(Context*, IParameter*)> slot([&param, &update_handler_called](Context* ctx, IParameter* param_in)
	{
		update_handler_called = param_in == &param;
	});
	param.RegisterUpdateNotifier(&slot);
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);
}


BOOST_AUTO_TEST_CASE(enum_params)
{
    mo::EnumParameter enum_param = {{"test", 5}};
    
}

BOOST_AUTO_TEST_CASE(input_parameter)
{
	int value = 10;
	TypedParameterPtr<int> param("Test wrapped param", &value);
	ITypedInputParameter<int> input_param;
	BOOST_REQUIRE(input_param.SetInput(&param));
	BOOST_REQUIRE_EQUAL(input_param.GetData(), value);
	
	bool update_handler_called = false;
	TypedSlot<void(Context*, IParameter*)> slot(
		[&update_handler_called](Context*, IParameter*)
	{
		update_handler_called = true;
	});

	BOOST_REQUIRE(input_param.RegisterUpdateNotifier(&slot));
	param.UpdateData(5);
	BOOST_REQUIRE_EQUAL(update_handler_called, true);
}

BOOST_AUTO_TEST_CASE(access_parameter)
{
    MetaObjectFactory::Instance()->RegisterTranslationUnit();

    auto obj = rcc::shared_ptr<parametered_object>::Create();
    obj->GetParameter<int>("int_value");
    obj->GetParameter<double>("double_value");
    BOOST_REQUIRE_EQUAL(obj->GetParameterValue<int>("int_value"), 0);
    obj->update(10);
    BOOST_REQUIRE_EQUAL(obj->GetParameterValue<int>("int_value"), 10);

}

