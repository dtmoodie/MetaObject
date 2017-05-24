
#define BOOST_TEST_MAIN
#define _VARIADIC_MAX  10
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
#include "MetaObject/params/NamedParam.hpp"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "Param"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;
struct PrintConstruct
{
    PrintConstruct()
    {
        std::cout << "construct\n";
    }
    int member = 0;
};



namespace test
{
    MO_KEYWORD_INPUT(timestamp, double)
    MO_KEYWORD_INPUT(frame_number, size_t)
    MO_KEYWORD_INPUT(dummy, PrintConstruct)
    MO_KEYWORD_OUTPUT(output, PrintConstruct)
    MO_KEYWORD_INPUT(optional, int)
}


template<class... Args>
void keywordFunction(const Args&... args)
{
    const size_t& fn = mo::GetKeywordInput<test::tag::frame_number>(args...);
    const double& timestamp = mo::GetKeywordInput<test::tag::timestamp>(args...);
    //const PrintConstruct& pc = mo::GetKeywordInput<test::tag::dummy>(args...);
    const int& optional = mo::GetKeywordInputDefault<test::tag::optional>(4, args...);
    std::cout << "Frame number: " << fn << "\n";
    std::cout << "Timestamp: " << timestamp << std::endl;
    std::cout << "Optional: " << optional << std::endl;
    std::cout << "Positional: " << mo::GetPositionalInput<0>(args...) << std::endl;
    std::cout << "size_t count: " << mo::CountType<size_t>(args...) << std::endl;
    PrintConstruct& pc_out = GetKeywordOutput<test::tag::output>(args...);
    pc_out.member = fn;
}


BOOST_AUTO_TEST_CASE(named_Param)
{
    size_t fn = 100;
    PrintConstruct pc;
    int asdf = 10;
    keywordFunction(asdf, test::tag::_frame_number = fn, test::tag::_timestamp = 0.5, test::tag::_dummy = pc, test::tag::_output = pc);
    fn = 200;
    keywordFunction(asdf, fn, test::tag::_dummy = pc, test::tag::_timestamp = 1.0, test::tag::_output = pc, test::tag::_optional = 5);

}


