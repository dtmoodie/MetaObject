#include "obj.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"

class ConcreteImplementation : public ExampleInterface
{
  public:
    MO_DERIVE(ConcreteImplementation, ExampleInterface)
    PARAM(int, integer_Param, 0)
    PARAM(float, float_Param, 0.0)
    INPUT(int, input_int_Param, nullptr)
    OUTPUT(int, output_int_Param, 0)
    MO_END
    static void PrintHelp() { std::cout << "Concrete PrintHelp() called\n"; }

    void foo() { std::cout << "Concrete implemtnation of foo called\n"; }
};

MO_REGISTER_CLASS(ConcreteImplementation)
