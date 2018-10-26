#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <MetaObject/params/TMultiInput-inl.hpp>
#include <MetaObject/params/TMultiOutput.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <iostream>

void printInputs(const std::tuple<const int*, const float*, const double*>& inputs)
{
    if (std::get<0>(inputs))
    {
        std::cout << "[int] " << *std::get<0>(inputs) << std::endl;
        return;
    }
    if (std::get<1>(inputs))
    {
        std::cout << "[float] " << *std::get<0>(inputs) << std::endl;
        return;
    }
    if (std::get<2>(inputs))
    {
        std::cout << "[double] " << *std::get<0>(inputs) << std::endl;
        return;
    }
    std::cout << "No input set" << std::endl;
}
BOOST_AUTO_TEST_CASE(polymorphic_input)
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
    BOOST_REQUIRE(multi_input.setInput(&int_out));

    BOOST_REQUIRE(multi_input.getInputParam());
    BOOST_REQUIRE_EQUAL(multi_input.getInputParam(), &int_out);
    int_out.updateData(6);
    BOOST_REQUIRE_NE(mo::get<const int*>(inputs), (void*)nullptr);
    BOOST_REQUIRE_EQUAL(*mo::get<const int*>(inputs), 6);
    printInputs(inputs);

    int_out.updateData(5);
    auto data = multi_input.getData(mo::Header());
    printInputs(inputs);

    mo::TMultiOutput<int, float, double> multi_output;
}
