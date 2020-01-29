#include "OutputParam.hpp"

namespace mo
{
    OutputParam::~OutputParam()
    {

    }

    bool OutputParam::providesOutput(const TypeInfo type) const
    {
        return type == getTypeInfo();
    }
}
