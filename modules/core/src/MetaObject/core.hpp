#pragma once
#define METAOBJECT_MODULE "core"
#include "MetaObject/detail/Export.hpp"
#undef METAOBJECT_MODULE
#include "MetaObject/core/Context.hpp"
#include "MetaObject/core/detail/Forward.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/logging/logging.hpp"

struct SystemTable;
namespace mo
{
    MO_EXPORTS void initCoreModule(SystemTable* table);
}
