#include <MetaObject/core/detail/Enums.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

void testFlags(const mo::ParamFlags flag)
{
    mo::EnumClassBitset<mo::ParamFlags> flags;
    flags.set(flag);
    BOOST_REQUIRE(flags.test(flag) == true);
    flags.reset(flag);
    BOOST_REQUIRE(flags.test(flag) == false);
}

BOOST_AUTO_TEST_CASE(enum_bitset)
{
    mo::EnumClassBitset<mo::ParamFlags> flags;

    BOOST_REQUIRE(flags.test(mo::ParamFlags::Input_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Output_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::State_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Control_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Buffer_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Optional_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Desynced_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Unstamped_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Sync_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::RequestBuffered_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Source_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Dynamic_e) == false);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::OwnsMutex_e) == false);

    testFlags(mo::ParamFlags::Input_e);
    testFlags(mo::ParamFlags::Output_e);
    testFlags(mo::ParamFlags::Control_e);
    testFlags(mo::ParamFlags::State_e);
    testFlags(mo::ParamFlags::Buffer_e);
    testFlags(mo::ParamFlags::Optional_e);
    testFlags(mo::ParamFlags::Desynced_e);
    testFlags(mo::ParamFlags::Unstamped_e);
    testFlags(mo::ParamFlags::Sync_e);
    testFlags(mo::ParamFlags::RequestBuffered_e);
    testFlags(mo::ParamFlags::Source_e);
    testFlags(mo::ParamFlags::Dynamic_e);
    testFlags(mo::ParamFlags::OwnsMutex_e);

    flags.set(mo::ParamFlags::Output_e);
    flags.set(mo::ParamFlags::Control_e);
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Output_e));
    BOOST_REQUIRE(flags.test(mo::ParamFlags::Control_e));
}
