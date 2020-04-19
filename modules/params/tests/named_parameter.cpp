#ifdef _MSC_VER
#define _VARIADIC_MAX 10
#endif

#include <MetaObject/params/NamedParam.hpp>
#include <MetaObject/params/ParamTags.hpp>
#include <MetaObject/params/TControlParam.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <mutex>

struct WidthTag
{
    using value_type = int;
    using storage_type = value_type;
    using pointer_type = storage_type*;
};
struct OptionalWidthTag
{
    using value_type = int;
    using storage_type = const value_type*;
    using pointer_type = storage_type;
};

static constexpr mo::TKeyword<WidthTag> width;
static constexpr mo::TKeyword<OptionalWidthTag> optional_width;

using Width = mo::TaggedValue<WidthTag>;
using OptionalWidth = mo::TaggedValue<OptionalWidthTag>;

int testFunc(Width w)
{
    return w;
}

TEST(named_parameter, simple_function)
{
    EXPECT_EQ(testFunc(Width(10)), 10);
}

TEST(named_parameter, simple_function_keyword)
{
    EXPECT_EQ(testFunc(width = 10), 10);
}

int testFunctOptionalInput(OptionalWidth val)
{
    const int* ptr = val;
    if (ptr)
    {
        return *ptr;
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
    auto foo = [](mo::tagged_values::Timestamp ts) -> mo::Time { return ts; };

    EXPECT_EQ(foo(mo::tags::timestamp = 10 * mo::ms), mo::Time(10 * mo::ms));
}

TEST(named_parameter, tag_framenumber)
{
    auto foo = [](mo::tagged_values::FrameNumber ts) -> uint64_t { return ts; };

    EXPECT_EQ(foo(mo::tags::fn = 10), 10);
}

TEST(named_parameter, tag_name)
{
    auto foo = [](mo::tagged_values::Name name) -> std::string { return name; };
    auto result = foo(mo::tags::name = "asdf");
    EXPECT_EQ(result, "asdf");
}

TEST(named_parameter, tag_param)
{
    auto foo = [](mo::tagged_values::Param param) -> std::string {
        const mo::IParam* p = param;
        EXPECT_TRUE(p);
        return p->getName();
    };
    mo::TControlParam<int> param;
    param.setName("asdf");

    EXPECT_EQ(foo(mo::tags::param = &param), "asdf");
}

template <class... Args>
int optionalInput(Args&&... args)
{
    const int* val = mo::getKeywordInputOptional<WidthTag>(std::forward<Args>(args)...);
    if (val)
    {
        return *val;
    }
    return 10;
}

TEST(named_parameter, optional_input)
{
    static_assert(mo::hasTaggedValue<float, int, double, Width>(), "");
    EXPECT_EQ(optionalInput(Width(10)), 10);
    EXPECT_EQ(optionalInput(4), 4);
    EXPECT_EQ(optionalInput(width = 15), 15);
    EXPECT_EQ(optionalInput(), 10);
}

template <bool STRICT = false, class... Ts>
mo::Header buildHeader(Ts&&... args)
{
    auto header = mo::getKeywordInputDefault<mo::tags::Header, STRICT>(mo::Header(), std::forward<Ts>(args)...);
    auto timestamp = mo::getKeywordInputOptional<mo::tags::Timestamp, STRICT>(std::forward<Ts>(args)...);
    auto fn = mo::getKeywordInputOptional<mo::tags::FrameNumber, STRICT>(std::forward<Ts>(args)...);
    auto src = mo::getKeywordInputOptional<mo::tags::Source, STRICT>(std::forward<Ts>(args)...);
    if (timestamp)
    {
        header.timestamp = *timestamp;
    }
    if (fn)
    {
        header.frame_number = *fn;
    }
    if (src)
    {
        header.source_id = *src;
    }
    return header;
}

TEST(named_parameter, tagged_header)
{
    auto header = buildHeader(mo::tagged_values::Header(mo::Header(5)));
    ASSERT_FALSE(header.timestamp);
    ASSERT_EQ(header.frame_number, 5);
}

TEST(named_parameter, keyword_header)
{
    auto header = buildHeader(mo::tags::header = mo::Header(5));
    ASSERT_FALSE(header.timestamp);
    ASSERT_EQ(header.frame_number, 5);
}

TEST(named_parameter, implicit_header)
{
    auto header = buildHeader(mo::Header(5));
    ASSERT_FALSE(header.timestamp);
    ASSERT_EQ(header.frame_number, 5);
}

TEST(named_parameter, explicit_header)
{
    auto header = buildHeader<true>(mo::Header(5));
    ASSERT_FALSE(header.timestamp);
    ASSERT_NE(header.frame_number, 5);
    ASSERT_FALSE(header.frame_number.valid());
}

// pass in timestamp
TEST(named_parameter, tagged_timestamp)
{
    auto header = buildHeader(mo::tagged_values::Timestamp(5 * mo::ms));
    ASSERT_TRUE(header.timestamp);
    ASSERT_FALSE(header.frame_number.valid());
    ASSERT_EQ(*header.timestamp, mo::Time(5 * mo::ms));
}

TEST(named_parameter, keyword_timestamp)
{
    auto header = buildHeader(mo::tags::timestamp = 5 * mo::ms);
    ASSERT_TRUE(header.timestamp);
    ASSERT_FALSE(header.frame_number.valid());
    ASSERT_EQ(*header.timestamp, mo::Time(5 * mo::ms));
}

TEST(named_parameter, implicit_timestamp)
{
    auto header = buildHeader(mo::Time(mo::ms * 5));
    ASSERT_TRUE(header.timestamp);
    ASSERT_FALSE(header.frame_number.valid());
    ASSERT_EQ(*header.timestamp, mo::Time(mo::ms * 5));
}

TEST(named_parameter, explicit_timestamp)
{
    auto header = buildHeader<true>(mo::ms * 5);
    ASSERT_FALSE(header.timestamp);
    ASSERT_FALSE(header.frame_number.valid());
}

