#include "MetaObject/core.hpp"
#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/detail/print_data.hpp"
#include <ostream>

struct NonPrintableStruct
{
    int a;
    int b;
    int c;
};

int main()
{
    static_assert(ct::StreamWritable<NonPrintableStruct>::value == false, "asdf");
    SystemTable table;
    mo::initCoreModule(&table);
    PerModuleInterface::GetInstance()->SetSystemTable(&table);
    mo::TParamPtr<NonPrintableStruct> param;
    NonPrintableStruct data;
    param.updatePtr(&data);
    param.print(std::cout);
    std::cout << std::endl;
}
