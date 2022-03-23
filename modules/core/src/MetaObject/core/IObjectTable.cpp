

#include "IObjectTable.hpp"

namespace mo
{
    // This is placed here to ensure the vtable is in this translation unit
    // not that it matters, but warnings and stuff
    IObjectTable::IObjectContainer::~IObjectContainer()
    {
    }
} // namespace mo
