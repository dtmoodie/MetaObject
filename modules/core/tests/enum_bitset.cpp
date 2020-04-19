#include <MetaObject/core/detail/Enums.hpp>

#include <ct/reflect_traits.hpp>
#include <ct/static_asserts.hpp>

#include "gtest/gtest.h"

void testFlags(const mo::ParamFlags flag)
{
    ct::EnumBitset<mo::ParamFlags> flags;
    flags.set(flag);
    ASSERT_EQ(flags.test(flag), true);
    flags.reset(flag);
    ASSERT_EQ(flags.test(flag), false);
}

template <uint64_t V>
mo::ParamFlags fromTemplateArg()
{
    return mo::ParamFlags(V);
}

TEST(enums, enum_bitset)
{
    static_assert(ct::IsReflected<mo::ParamFlags>::value, "asdf");
    using type = ct::PtrType<mo::ParamFlags, 0>;
    static_assert(ct::IsEnumField<type>::value, "asdf");
    static_assert(ct::EnumChecker<mo::ParamFlags>::value, "asdf");
    ct::StaticEquality<uint64_t, ct::Flags::CT_RESERVED_FLAG_BITS + 1, 9>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kCONTROL, 1 << 9>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kSTATE, 1 << 10>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kSTATUS, 1 << 11>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kINPUT, 1 << 12>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kOUTPUT, 1 << 13>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kOPTIONAL, 1 << 14>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kSOURCE, 1 << 15>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kSIGNAL, 1 << 16>{};
    ct::StaticEquality<uint64_t, mo::ParamReflectionFlags::kSLOT, 1 << 17>{};

    ct::EnumBitset<mo::ParamFlags> flags;

    EXPECT_FALSE(flags.test(mo::ParamFlags::kINPUT));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kOUTPUT));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kSTATE));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kCONTROL));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kBUFFER));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kOPTIONAL));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kDESYNCED));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kUNSTAMPED));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kSYNC));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kREQUIRE_BUFFERED));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kSOURCE));
    EXPECT_FALSE(flags.test(mo::ParamFlags::kDYNAMIC));

    testFlags(mo::ParamFlags::kINPUT);
    testFlags(mo::ParamFlags::kOUTPUT);
    testFlags(mo::ParamFlags::kCONTROL);
    testFlags(mo::ParamFlags::kSTATE);
    testFlags(mo::ParamFlags::kBUFFER);
    testFlags(mo::ParamFlags::kOPTIONAL);
    testFlags(mo::ParamFlags::kDESYNCED);
    testFlags(mo::ParamFlags::kUNSTAMPED);
    testFlags(mo::ParamFlags::kSYNC);
    testFlags(mo::ParamFlags::kREQUIRE_BUFFERED);
    testFlags(mo::ParamFlags::kSOURCE);
    testFlags(mo::ParamFlags::kDYNAMIC);

    flags.set(mo::ParamFlags::kOUTPUT);
    flags.set(mo::ParamFlags::kCONTROL);
    ASSERT_EQ(flags.test(mo::ParamFlags::kOUTPUT), true);
    ASSERT_EQ(flags.test(mo::ParamFlags::kCONTROL), true);

    auto bitset = fromTemplateArg<mo::ParamFlags::kOUTPUT>();
    std::cout << bitset << std::endl;
    ASSERT_EQ(bitset.test(mo::ParamFlags::kOUTPUT), true);
    bitset = fromTemplateArg<mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE>();

    for (size_t i = 0; i < 64; ++i)
    {
        if (i != mo::ParamFlags::kOUTPUT && i != mo::ParamFlags::kSOURCE)
        {
            uint64_t val = 1;
            val = val << i;
            ASSERT_EQ(bitset.test(val), false);
        }
    }

    ASSERT_EQ(bitset.test(mo::ParamFlags::kOUTPUT), true);
    ASSERT_EQ(bitset.test(mo::ParamFlags::kSOURCE), true);
    auto test = mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE | mo::ParamFlags::kREQUIRE_BUFFERED;

    EXPECT_TRUE(test.test(mo::ParamFlags::kOUTPUT));
    EXPECT_TRUE(test.test(mo::ParamFlags::kSOURCE));
    EXPECT_TRUE(test.test(mo::ParamFlags::kREQUIRE_BUFFERED));

    bitset = fromTemplateArg<mo::ParamFlags::kOUTPUT | mo::ParamFlags::kSOURCE | mo::ParamFlags::kREQUIRE_BUFFERED>();

    for (size_t i = 0; i < 64; ++i)
    {
        if (i != mo::ParamFlags::kOUTPUT && i != mo::ParamFlags::kSOURCE && i != mo::ParamFlags::kREQUIRE_BUFFERED)
        {
            uint64_t val = 1;
            val = val << i;
            ASSERT_FALSE(bitset.test(val));
        }
    }
    ASSERT_EQ(bitset.test(mo::ParamFlags::kOUTPUT), true);
    ASSERT_EQ(bitset.test(mo::ParamFlags::kSOURCE), true);
    ASSERT_EQ(bitset.test(mo::ParamFlags::kREQUIRE_BUFFERED), true);
}
