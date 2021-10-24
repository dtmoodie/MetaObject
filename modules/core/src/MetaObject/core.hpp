#pragma once
#include <MetaObject/core/export.hpp>

#include <MetaObject/core/AsyncStream.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/logging/logging.hpp>

struct SystemTable;
namespace mo
{
    MO_EXPORTS void initCoreModule(SystemTable* table);
}
