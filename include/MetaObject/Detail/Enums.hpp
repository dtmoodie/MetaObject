#pragma once
#include "Export.hpp"
#include <string>
namespace mo
{
    enum ParameterType
    {
        None_e = 0,
        Input_e = 1,
        Output_e = 2,
        State_e = 4,
        Control_e = 8,
        Buffer_e = 16,
        Optional_e = 32,
        Desynced_e = 64,
        Constant_e = 128 /* If this flag is set, the timestamp will not be set on this parameter */
    };
    MO_EXPORTS std::string ParameteTypeToString(ParameterType type);
    MO_EXPORTS ParameterType StringToParameteType(const std::string& str);
    enum ParameterTypeFlags
    {
        TypedParameter_e = 0,
        CircularBuffer_e ,
        ConstMap_e,
        Map_e,
        StreamBuffer_e,
        BlockingStreamBuffer_e,
        NNStreamBuffer_e,
        Queue_e,
        ForceBufferedConnection_e = 1024,
        ForceDirectConnection_e = 2048
    };
    MO_EXPORTS std::string ParameterTypeFlagsToString(ParameterTypeFlags flags);
    MO_EXPORTS ParameterTypeFlags StringToParameterTypeFlags(const std::string& str);

}
