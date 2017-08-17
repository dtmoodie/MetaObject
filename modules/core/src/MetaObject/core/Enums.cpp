#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/logging/logging.hpp"
#include <vector>
using namespace mo;

#define TYPE_NAME_HELPER(name) {ParamFlags::name##_e, #name }

static const std::vector<std::pair<ParamFlags, std::string>> type_flag_map = {
    TYPE_NAME_HELPER(None),
    TYPE_NAME_HELPER(Input),
    TYPE_NAME_HELPER(Output),
    TYPE_NAME_HELPER(State),
    TYPE_NAME_HELPER(Control),
    TYPE_NAME_HELPER(Buffer),
    TYPE_NAME_HELPER(Optional),
    TYPE_NAME_HELPER(Unstamped),
    TYPE_NAME_HELPER(Source),
    TYPE_NAME_HELPER(Sync),
    TYPE_NAME_HELPER(RequestBuffered),
    TYPE_NAME_HELPER(Dynamic),
    TYPE_NAME_HELPER(Source)
};

std::string mo::paramFlagsToString(EnumClassBitset<ParamFlags> type) {
    std::string output;
    for(const auto& itr : type_flag_map){
        if(type.test(itr.first)){
            if(output.empty()){
                output = itr.second;
            }else{
                output += "|" + itr.second;
            }
        }
    }
    return output;
}

EnumClassBitset<ParamFlags> mo::stringToParamFlags(const std::string& str) {
    std::string rest = str;
    EnumClassBitset<ParamFlags> output;
    auto pos = rest.find('|');
    while(pos != std::string::npos){
        std::string substr = rest.substr(0, pos);
        rest = rest.substr(pos + 1);
        for(const auto& itr : type_flag_map){
            if(substr == itr.second)
                //output = ParamFlags(itr.first | output);
                output.flip(itr.first);
        }
        pos = rest.find('|');
    }
    for(const auto& itr : type_flag_map){
        if(rest == itr.second)
            //output = ParamFlags(itr.first | output);
            output.flip(itr.first);
    }
    return output;
}

std::string mo::paramTypeToString(ParamType flags) {
    switch(flags) {
    case TParam_e:
        return "T";
    case CircularBuffer_e:
        return "circularbuffer";
    case ConstMap_e:
        return "constmap";
    case Map_e:
        return "map";
    case StreamBuffer_e:
        return "StreamBuffer";
    case BlockingStreamBuffer_e:
        return "BlockingStreamBuffer";
    case NNStreamBuffer_e:
        return "NNStreamBuffer";
    case Queue_e:
        return "Queue";
    case BlockingQueue_e:
        return "BlockingQueue";
    case DroppingQueue_e:
        return "DroppingQueue";
    case ForceDirectConnection_e:
        return "ForceDirectConnection";
    case ForceBufferedConnection_e:
        return "ForceBufferedConnection";
    }
    return "";
}

ParamType mo::stringToParamType(const std::string& str) {
    if(str == "T")
        return TParam_e;
    else if(str == "circularbuffer")
        return CircularBuffer_e;
    else if(str == "constmap")
        return ConstMap_e;
    else if(str == "map")
        return Map_e;
    else if(str == "StreamBuffer")
        return StreamBuffer_e;
    else if(str == "BlockingStreamBuffer")
        return BlockingStreamBuffer_e;
    else if(str == "NNStreamBuffer")
        return NNStreamBuffer_e;
    THROW(debug) << "Invalid string " << str;
    return TParam_e;
}
