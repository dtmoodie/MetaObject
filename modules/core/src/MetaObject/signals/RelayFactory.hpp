#pragma once
#include "MetaObject/detail/Export.hpp"
#include <functional>

namespace mo {
class TypeInfo;
class ISignalRelay;
class MO_EXPORTS RelayFactory {
public:
    static RelayFactory* instance();
    void registerCreator(std::function<ISignalRelay*(void)> f, const TypeInfo& type);
    ISignalRelay* create(const TypeInfo& type);

private:
    RelayFactory();
    ~RelayFactory();
    struct impl;
    impl* _pimpl = nullptr;
};
}
