#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "instantiate.hpp"


INSTANTIATE_META_PARAMETER(int);
INSTANTIATE_META_PARAMETER(ushort);
INSTANTIATE_META_PARAMETER(uint);
INSTANTIATE_META_PARAMETER(char);
INSTANTIATE_META_PARAMETER(uchar);
INSTANTIATE_META_PARAMETER(float);
INSTANTIATE_META_PARAMETER(double);


void mo::instantiations::initialize()
{
    
}