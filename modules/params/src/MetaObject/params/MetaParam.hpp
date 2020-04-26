#pragma once
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/core/detail/HelperMacros.hpp"
#include <typeinfo>

namespace mo
{
    template <class T, int N, typename Enable = void>
    struct MetaParam : public MetaParam<T, N - 1, void>
    {
        MetaParam(SystemTable* table, const char* name = nullptr)
            : MetaParam<T, N - 1>(table, name)
        {
        }
    };

    template <class T>
    struct MetaParam<T, 0, void>
    {
        MetaParam(SystemTable* table, const char* name = nullptr)
        {
            if (name)
            {
                mo::TypeTable::instance(table)->registerType(mo::TypeInfo::create<T>(), name);
            }
        }
    };
} // namespace mo
