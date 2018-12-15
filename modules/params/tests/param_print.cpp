#include "MetaObject/core.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/params.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/print_data.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <ostream>

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

struct GlobalFixture
{
    std::shared_ptr<SystemTable> table;

    GlobalFixture()
    {
        table = SystemTable::instance();
        PerModuleInterface::GetInstance()->SetSystemTable(table.get());
        mo::params::init(table.get());
    }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);

struct NonPrintableStruct
{
    int a;
    int b;
    int c;
};

BOOST_AUTO_TEST_CASE(non_printable)
{

    /*static_assert(ct::StreamWritable<NonPrintableStruct>::value == false, "asdf");
    mo::TParamPtr<NonPrintableStruct> param;
    NonPrintableStruct data;
    param.updatePtr(&data);
    param.print(std::cout);*/
}
