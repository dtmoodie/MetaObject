#include "Objects.hpp"
#include <MetaObject/object/RelayManager.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

using namespace test;

template <class SIG>
void testGetSignal(const std::string& name, IMetaObject* obj)
{
    auto signal = obj->getSignal(name, TypeInfo(typeid(SIG)));
    BOOST_REQUIRE(signal);
    BOOST_REQUIRE(dynamic_cast<TSignal<SIG>*>(signal));
}

BOOST_AUTO_TEST_CASE(signal_init_and_access)
{
    auto obj = DerivedSignals::create();
    BOOST_REQUIRE(obj);
    testGetSignal<void(int)>("base_signal", obj.get());
    testGetSignal<void(int)>("derived_signal", obj.get());
    testGetSignal<void(IMetaObject*, IParam*)>("param_added", obj.get());
    testGetSignal<void(IMetaObject*, IParam*)>("param_updated", obj.get());

    auto all_sigs = obj->getSignals();
    BOOST_REQUIRE_EQUAL(all_sigs.size(), 4);
}

BOOST_AUTO_TEST_CASE(signal_reception)
{
    int value = 0;
    mo::TSlot<void(int)> m_slot([&value](int val) { value = val; });
    auto obj = DerivedSignals::create();
    BOOST_REQUIRE(obj);

    RelayManager mgr;
    BOOST_REQUIRE_EQUAL(obj->setupSignals(&mgr), 8);
    auto connection = mgr.connect(&m_slot, "base_signal", nullptr);
    BOOST_REQUIRE(connection);

    obj->sig_base_signal(10);
    BOOST_REQUIRE_EQUAL(value, 10);
}
