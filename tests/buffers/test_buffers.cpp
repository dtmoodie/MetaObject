#define BOOST_TEST_MAIN
#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/buffers/IBuffer.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/types/file_types.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/thread.hpp>
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

#if BOOST_VERSION > 105800
#define MY_BOOST_TEST_ADD_ARGS __FILE__, __LINE__,
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR , boost::unit_test::decorator::collector::instance()
#else
#define MY_BOOST_TEST_ADD_ARGS
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR
#endif

#define BOOST_FIXTURE_PARAM_TEST_CASE(test_name, F, mbegin, mend)                                                      \
    struct test_name : public F                                                                                        \
    {                                                                                                                  \
        typedef ::std::remove_const<::std::remove_reference<decltype(*(mbegin))>::type>::type param_t;                 \
        void test_method(const param_t&);                                                                              \
    };                                                                                                                 \
                                                                                                                       \
    void BOOST_AUTO_TC_INVOKER(test_name)(const test_name::param_t& param)                                             \
    {                                                                                                                  \
        test_name t;                                                                                                   \
        t.test_method(param);                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    BOOST_AUTO_TU_REGISTRAR(test_name)                                                                                 \
    (boost::unit_test::make_test_case(                                                                                 \
        &BOOST_AUTO_TC_INVOKER(test_name), #test_name, MY_BOOST_TEST_ADD_ARGS(mbegin), (mend))                         \
         MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR);                                                                         \
                                                                                                                       \
    void test_name::test_method(const param_t& param)

#define BOOST_AUTO_PARAM_TEST_CASE(test_name, mbegin, mend)                                                            \
    BOOST_FIXTURE_PARAM_TEST_CASE(test_name, BOOST_AUTO_TEST_CASE_FIXTURE, mbegin, mend)

template <typename T, size_t sz>
size_t size(T (&)[sz])
{
    return sz;
}

template <typename T, size_t sz>
T* end(T (&ptr)[sz])
{
    return ptr + sz;
}

struct GlobalFixture
{
    ~GlobalFixture()
    {
        // mo::ThreadPool::instance()->cleanup();
        // mo::ThreadSpecificQueue::cleanup();
    }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
struct BufferFixture
{
    BufferFixture()
    {
        stream = mo::AsyncStreamFactory::instance()->create();
        output_param.updatePtr(&output);
        input_param.setUserDataPtr(&input);
        output_param.setStream(stream.get());
        input_param.setStream(stream.get());
    }
    mo::TParamPtr<int> output_param;
    int output;
    mo::TInputParamPtr<int> input_param;
    const int* input;
    mo::IAsyncStream::Ptr_t stream;
};

BOOST_FIXTURE_TEST_SUITE(buffer_suite, BufferFixture)

static const mo::BufferFlags buffer_test_cases[] = {
    mo::CIRCULAR_BUFFER, mo::MAP_BUFFER, mo::STREAM_BUFFER, mo::BLOCKING_STREAM_BUFFER, mo::NEAREST_NEIGHBOR_BUFFER};

BOOST_AUTO_PARAM_TEST_CASE(buffer_test, buffer_test_cases, end(buffer_test_cases))
{
    std::cout << "Testing " << mo::bufferFlagsToString(param) << std::endl;
    auto buffer = std::shared_ptr<mo::IParam>(mo::buffer::BufferFactory::createBuffer(&output_param, param));
    BOOST_REQUIRE(buffer);
    auto buf = std::dynamic_pointer_cast<mo::buffer::IBuffer>(buffer);
    BOOST_REQUIRE(buf);
    buf->setFrameBufferCapacity(100);
    input_param.setInput(buffer);
    std::vector<mo::Time> process_queue;
    mo::UpdateSlot_t slot([&process_queue](mo::IParam*, mo::Header hdr, mo::UpdateFlags fg) {
        if (fg == mo::UpdateFlags::BufferUpdated_e)
            process_queue.push_back(*hdr.timestamp);
    });
    auto connection = input_param.registerUpdateNotifier(&slot);
    for (int j = 0; j < 5; ++j)
    {
        for (int i = 50 * j; i < 50 + 50 * j; ++i)
        {
            output_param.updateData(i, mo::tag::_timestamp = mo::Time(i * mo::ms));
        }
        for (auto itr = process_queue.begin(); itr != process_queue.end();)
        {
            auto container = input_param.getTypedData<int>(*itr);
            BOOST_REQUIRE(container);
            BOOST_REQUIRE_EQUAL(mo::Time(container->data * mo::ms), *itr);
            itr = process_queue.erase(itr);
        }
    }

    for (int j = 0; j < 5; ++j)
    {
        for (int i = 50 * j; i < 50 + 50 * j; ++i)
        {
            output = i * 2;
            output_param.emitUpdate(mo::Header(mo::Time(i * mo::ms)));
        }
        for (auto itr = process_queue.begin(); itr != process_queue.end();)
        {
            auto container = input_param.getTypedData<int>(*itr);
            BOOST_REQUIRE(container);
            BOOST_REQUIRE_EQUAL(mo::Time((container->data / 2) * mo::ms), *itr);
            itr = process_queue.erase(itr);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
