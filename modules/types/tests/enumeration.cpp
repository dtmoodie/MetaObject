#include <MetaObject/types/file_types.hpp>

#include <ct/reflect_macros.hpp>

#include <gtest/gtest.h>

namespace
{
    ENUM_BEGIN(PixelFormat, uint8_t)
        ENUM_VALUE(kUNCHANGED, 0)
        ENUM_VALUE(kGRAY, 1)
        ENUM_VALUE(kRGB, kGRAY + 1)
        ENUM_VALUE(kBGR, kRGB + 1)
        ENUM_VALUE(kHSV, kRGB + 1)
        ENUM_VALUE(kHSL, kHSV + 1)
        ENUM_VALUE(kLUV, kHSL + 1)
        ENUM_VALUE(kARGB, kLUV + 1)
        ENUM_VALUE(kRGBA, kARGB + 1)
        ENUM_VALUE(kBGRA, kRGBA + 1)
        ENUM_VALUE(kABGR, kBGRA + 1)
        ENUM_VALUE(kBAYER_BGGR, kABGR + 1)
        ENUM_VALUE(kBAYER_RGGB, kBAYER_BGGR + 1)

        constexpr uint8_t numChannels() const;
    ENUM_END;
} // namespace

TEST(enumeration, from_reflection)
{
    mo::EnumParam enum_param{PixelFormat()};
    ASSERT_EQ(enum_param.enumerations.size(), 13);
    ASSERT_EQ(enum_param.enumerations[0], "kUNCHANGED");
    ASSERT_EQ(enum_param.enumerations[1], "kGRAY");
    ASSERT_EQ(enum_param.enumerations[2], "kRGB");
    ASSERT_EQ(enum_param.enumerations[3], "kBGR");
    ASSERT_EQ(enum_param.enumerations[4], "kHSV");
    ASSERT_EQ(enum_param.enumerations[5], "kHSL");
    ASSERT_EQ(enum_param.enumerations[6], "kLUV");
    ASSERT_EQ(enum_param.enumerations[7], "kARGB");
    ASSERT_EQ(enum_param.enumerations[8], "kRGBA");
    ASSERT_EQ(enum_param.enumerations[9], "kBGRA");
    ASSERT_EQ(enum_param.enumerations[10], "kABGR");
    ASSERT_EQ(enum_param.enumerations[11], "kBAYER_BGGR");
    ASSERT_EQ(enum_param.enumerations[12], "kBAYER_RGGB");
}
