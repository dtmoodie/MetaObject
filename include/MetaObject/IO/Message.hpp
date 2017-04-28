#pragma once
#include <string>
#include <map>
#include "RuntimeObjectSystem/ObjectInterface.h"
namespace mo {
class IParam;
class IMetaObject;

struct Message {
    std::string topic;
    std::map<ObjectId, IMetaObject*> objects;
    std::map<std::string, IParam*> Params;
    template<class AR> void serialize(AR& ar) {
        ar(topic);
        ar(objects);
        ar(Params);
    }
};
struct ParamUpdate {

};
}