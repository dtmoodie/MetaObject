#define BOOST_TEST_MAIN
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/Detail/SignalMacros.hpp"
#include "MetaObject/Signals/Detail/SlotMacros.hpp"
#include "MetaObject/Params/ParamMacros.hpp"
#include "MetaObject/Params/TParamPtr.hpp"
#include "MetaObject/Params/TInputParam.hpp"
#include "MetaObject/Params/Types.hpp"
#include "MetaObject/Params/Buffers/BufferFactory.hpp"
#include "MetaObject/Params/Buffers/IBuffer.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
#include "MetaObject/Detail/Allocator.hpp"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include <boost/any.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/common.hpp>
#include <boost/log/attributes.hpp>
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>


#if BOOST_VERSION > 105800
#define MY_BOOST_TEST_ADD_ARGS __FILE__, __LINE__,
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR ,boost::unit_test::decorator::collector::instance()
#else
#define MY_BOOST_TEST_ADD_ARGS
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR
#endif

#define BOOST_FIXTURE_PARAM_TEST_CASE( test_name, F, mbegin, mend )     \
struct test_name : public F {                                           \
   typedef ::std::remove_const< ::std::remove_reference< decltype(*(mbegin)) >::type>::type param_t; \
   void test_method(const param_t &);                                   \
};                                                                      \
                                                                        \
void BOOST_AUTO_TC_INVOKER( test_name )(const test_name::param_t &param) \
{                                                                       \
    test_name t;                                                        \
    t.test_method(param);                                               \
}                                                                       \
                                                                        \
BOOST_AUTO_TU_REGISTRAR( test_name )(                                   \
    boost::unit_test::make_test_case(                                   \
       &BOOST_AUTO_TC_INVOKER( test_name ), #test_name,                 \
       MY_BOOST_TEST_ADD_ARGS                                           \
       (mbegin), (mend))                                                \
       MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR);                            \
                                                                        \
void test_name::test_method(const param_t &param)                       \

#define BOOST_AUTO_PARAM_TEST_CASE( test_name, mbegin, mend )           \
   BOOST_FIXTURE_PARAM_TEST_CASE( test_name,                            \
                                  BOOST_AUTO_TEST_CASE_FIXTURE,         \
                                  mbegin, mend)

template<typename T, size_t sz>
size_t size(T(&)[sz]) {
    return sz;
}

template<typename T, size_t sz>
T* end(T(&ptr)[sz]) {
    return ptr + sz;
}

struct GlobalFixture{
    ~GlobalFixture(){
        mo::ThreadPool::Instance()->Cleanup();
        mo::ThreadSpecificQueue::Cleanup();
        mo::Allocator::CleanupThreadSpecificAllocator();
    }
};
BOOST_GLOBAL_FIXTURE(GlobalFixture);
struct BufferFixture{
    BufferFixture(){
        output_param.updatePtr(&output);
        input_param.setUserDataPtr(&input);
        output_param.setContext(mo::Context::GetDefaultThreadContext());
        input_param.setContext(mo::Context::GetDefaultThreadContext());
    }
    mo::TParamPtr<int> output_param;
    int output;
    mo::TInputParamPtr<int> input_param;
    const int* input;
};

BOOST_FIXTURE_TEST_SUITE(buffer_suite, BufferFixture)

static const mo::ParamType buffer_test_cases[] = {
    mo::CircularBuffer_e, mo::Map_e, mo::StreamBuffer_e, mo::BlockingStreamBuffer_e, mo::NNStreamBuffer_e
};

BOOST_AUTO_PARAM_TEST_CASE(buffer_test, buffer_test_cases, end(buffer_test_cases)){
    std::cout << "Testing " << mo::paramTypeToString(param) << std::endl;
    auto buffer = mo::Buffer::BufferFactory::CreateProxy(&output_param, param);
    BOOST_REQUIRE(buffer);
    auto buf = std::dynamic_pointer_cast<mo::Buffer::IBuffer>(buffer);
    BOOST_REQUIRE(buf);
    buf->setFrameBufferCapacity(100);
    input_param.setInput(buffer);
    std::vector<mo::Time_t> process_queue;
    mo::UpdateSlot_t slot([&process_queue](mo::IParam* param, mo::Context* ctx, mo::OptionalTime_t ts, size_t fn, mo::ICoordinateSystem* cs, mo::UpdateFlags fg){
        if(fg == mo::UpdateFlags::BufferUpdated_e)
            process_queue.push_back(*ts);
    });
    auto connection = input_param.registerUpdateNotifier(&slot);
    for(int j = 0; j < 5; ++j){
        for (int i = 50 * j; i < 50 + 50 * j; ++i) {
            output_param.updateData(i, mo::tag::_timestamp = mo::Time_t(i * mo::ms));
        }
        for (auto itr = process_queue.begin(); itr != process_queue.end();) {
            int data;
            BOOST_REQUIRE(input_param.getData(data, *itr));
            BOOST_REQUIRE_EQUAL(mo::Time_t(data * mo::ms), *itr);
            itr = process_queue.erase(itr);
        }
    }

    for(int j = 0; j < 5; ++j){
        for(int i = 50 * j; i < 50 + 50 * j; ++i){
            output = i * 2;
            output_param.emitUpdate(mo::OptionalTime_t(i * mo::ms));
        }
        for (auto itr = process_queue.begin(); itr != process_queue.end();) {
            int data;
            BOOST_REQUIRE(input_param.getData(data, *itr));
            BOOST_REQUIRE_EQUAL(mo::Time_t((data / 2) * mo::ms), *itr);
            itr = process_queue.erase(itr);
        }
    }

}

BOOST_AUTO_TEST_SUITE_END()
