#ifndef META_OBJECT_PARAMS_MODULE_HEADER
#define META_OBJECT_PARAMS_MODULE_HEADER
#define METAOBJECT_MODULE "params"
#include "MetaObject/detail/Export.hpp"
#undef METAOBJECT_MODULE
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#endif

namespace mo
{
    namespace params
    {
        void MO_EXPORTS init(SystemTable* table);
    }
} // namespace mo
