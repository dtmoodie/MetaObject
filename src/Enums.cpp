#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Logging/Log.hpp"
using namespace mo;
std::string mo::ParameteTypeToString(ParameterType type)
{
    switch(type)
    {
    case None_e: return "None";
    case Input_e: return "Input";
    case Output_e: return "Output";
    case State_e: return "State";
    case Control_e: return "Control";
    case Buffer_e: return "Buffer";
    case Optional_e: return "Optional";
    }
    return "";
}

ParameterType mo::StringToParameteType(const std::string& str)
{
    if(str == "None")
        return None_e;
    else if(str == "Input")
        return Input_e;
    else if(str == "Output")
        return Output_e;
    else if(str == "State")
        return State_e;
    else if(str == "Control")
        return Control_e;
    else if(str == "Buffer")
        return Buffer_e;
    else if(str == "Optional")
        return Optional_e;
    THROW(debug) << "Invalid string " << str;
}

std::string mo::ParameterTypeFlagsToString(ParameterTypeFlags flags)
{
    switch(flags)
    {
    case TypedParameter_e: return "Typed";
    case cbuffer_e: return "cbuffer";
    case cmap_e: return "cmap";
    case map_e: return "map";
    case StreamBuffer_e: return "StreamBuffer";
    case BlockingStreamBuffer_e: return "BlockingStreamBuffer";
    }
    return "";
}

ParameterTypeFlags mo::StringToParameterTypeFlags(const std::string& str)
{
    if(str == "Typed")
        return TypedParameter_e;
    else if(str == "cbuffer")
        return cbuffer_e;
    else if(str == "cmap")
        return cmap_e;
    else if(str == "map")
        return map_e;
    else if(str == "StreamBuffer")
        return StreamBuffer_e;
    else if(str == "BlockingStreamBuffer")
        return BlockingStreamBuffer_e;
    THROW(debug) << "Invalid string " << str;
}