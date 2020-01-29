#ifdef _MSC_VER
#define _VARIADIC_MAX 10
#endif

#include <MetaObject/params/NamedParam.hpp>
#include <MetaObject/params/ParamTags.hpp>
#include <MetaObject/params/TParam.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace mo;

struct WidthParam;
using Width = mo::TNamedParam<WidthParam, int>;
static constexpr mo::TKeyword<Width> width;

int testFunc(Width w)
{
    return w.get();
}

TEST(named_parameter, simple_function)
{
    EXPECT_EQ(testFunc(Width(10)), 10);
}

TEST(named_parameter, simple_function_keyword)
{
    EXPECT_EQ(testFunc(width = 10), 10);
}

struct OptionalWidthParam;
using OptionalWidth = mo::TNamedParam<OptionalWidthParam, const int*, const int*>;
static constexpr mo::TKeyword<OptionalWidth> optional_width;

int testFunctOptionalInput(OptionalWidth val)
{
    if (val.get())
    {
        return *val.get();
    }
    return 10;
}

TEST(named_parameter, optional_input_default)
{
    EXPECT_EQ(testFunctOptionalInput(OptionalWidth(nullptr)), 10);
}

TEST(named_parameter, optional_input_set)
{
    int val = 5;
    EXPECT_EQ(testFunctOptionalInput(OptionalWidth(&val)), 5);
}

TEST(named_parameter, optional_input_boost_optional)
{
    boost::optional<int> val;
    EXPECT_EQ(testFunctOptionalInput(OptionalWidth(val)), 10);
    val = 4;

    EXPECT_EQ(testFunctOptionalInput(OptionalWidth(val)), 4);
}

TEST(named_parameter, optional_input_keyword)
{
    int val = 5;
    EXPECT_EQ(testFunctOptionalInput(optional_width = &val), 5);
}

TEST(named_parameter, tag_timestamp)
{
    auto foo = [](mo::params::Timestamp ts) -> mo::Time { return ts; };

    EXPECT_EQ(foo(mo::timestamp = 10 * mo::ms), mo::Time(10 * mo::ms));
}

TEST(named_parameter, tag_framenumber)
{
    auto foo = [](mo::params::FrameNumber ts) -> uint64_t { return ts; };

    EXPECT_EQ(foo(mo::fn = 10), 10);
}

TEST(named_parameter, tag_name)
{
    auto foo = [](mo::params::Name name) -> std::string { return name; };
    auto result = foo(mo::name = "asdf");
    EXPECT_EQ(result, "asdf");
}

TEST(named_parameter, tag_param)
{
    auto foo = [](mo::params::Param param) -> std::string { return param.get()->getName(); };
    mo::TParam<int> param;
    param.setName("asdf");

    EXPECT_EQ(foo(mo::param = &param), "asdf");
}

template <class... Args>
int optionalInput(Args&&... args)
{
    const int* val = mo::getKeywordInputOptional<Width>(std::forward<Args>(args)...);
    if (val)
    {
        return *val;
    }
    return 10;
}

TEST(named_parameter, optional_input)
{
    static_assert(mo::hasNamedParam<float, int, double, Width>(), "");
    EXPECT_EQ(optionalInput(Width(10)), 10);
    EXPECT_EQ(optionalInput(4), 10);
    EXPECT_EQ(optionalInput(width = 15), 15);
    EXPECT_EQ(optionalInput(), 10);
}

template <class... Args>
void paramUpdate(int, Args&&... args)
{
    const IParam* param =
        getKeywordInputDefault<params::Param>(static_cast<const IParam*>(nullptr), std::forward<Args>(args)...);
    const uint64_t* fnptr = getKeywordInputOptional<params::FrameNumber>(std::forward<Args>(args)...);
    auto tsptr = getKeywordInputOptional<params::Timestamp>(std::forward<Args>(args)...);
    auto stream_ = getKeywordInputDefault<params::Stream>(nullptr, std::forward<Args>(args)...);
}

TEST(named_parameter, param_update)
{
    int timestamp = 1;

    paramUpdate(timestamp * 10, mo::timestamp = mo::ms * timestamp, mo::stream = nullptr);
}
