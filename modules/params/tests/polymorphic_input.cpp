#include <MetaObject/params/TMultiInput.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <iostream>

void printInputs(const std::tuple<const int*, const float*, const double*>& inputs)
{
    if (std::get<0>(inputs))
    {
        std::cout << "[int] " << *std::get<0>(inputs) << std::endl;
        ;
        return;
    }
    if (std::get<1>(inputs))
    {
        std::cout << "[float] " << *std::get<0>(inputs) << std::endl;
        ;
        return;
    }
    if (std::get<2>(inputs))
    {
        std::cout << "[double] " << *std::get<0>(inputs) << std::endl;
        ;
        return;
    }
    std::cout << "No input set" << std::endl;
}

int main(int argc, char** argv)
{
    std::tuple<const int*, const float*, const double*> inputs;
    int int_val;
    mo::TParamOutput<int> int_out;
    int_out.updatePtr(&int_val);

    float float_val;
    mo::TParamOutput<float> float_out;
    float_out.updatePtr(&float_val);

    double double_val;
    mo::TParamOutput<double> double_out;
    double_out.updatePtr(&double_val);

    printInputs(inputs);
    mo::TMultiInput<int, float, double> multi_input;
    mo::Mutex_t mtx;
    multi_input.setMtx(&mtx);
    multi_input.setUserDataPtr(&inputs);

    multi_input.setInput(&int_out);
    printInputs(inputs);
    int_out.updateData(5);
    multi_input.getInput(mo::OptionalTime_t(), nullptr);
    printInputs(inputs);
    return 0;
}
