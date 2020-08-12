#include "MetaObject/core.hpp"
#include "MetaObject/core/SystemTable.hpp"

#include "MetaObject/params/detail/print_data.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <ostream>

#include <gtest/gtest.h>

struct NonPrintableStruct
{
    int a;
    int b;
    int c;
};

TEST(param_print, non_printable)
{

    /*static_assert(ct::StreamWritable<NonPrintableStruct>::value == false, "asdf");
    mo::TParamPtr<NonPrintableStruct> param;
    NonPrintableStruct data;
    param.updatePtr(&data);
    param.print(std::cout);*/
}
