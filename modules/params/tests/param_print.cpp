#include <boost/test/unit_test_suite.hpp>

#include "MetaObject/core.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/params.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/print_data.hpp"

#include <boost/thread/recursive_mutex.hpp>
#include <ostream>

struct GlobalFixture
{
    SystemTable table;

    GlobalFixture()
    {
        PerModuleInterface::GetInstance()->SetSystemTable(&table);
        mo::params::init(&table);
    }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture)

struct NonPrintableStruct
{
    int a;
    int b;
    int c;
};

BOOST_AUTO_TEST_CASE(non_printable)
{

    static_assert(ct::StreamWritable<NonPrintableStruct>::value == false, "asdf");
    mo::TParamPtr<NonPrintableStruct> param;
    NonPrintableStruct data;
    param.updatePtr(&data);
    param.print(std::cout);
}
