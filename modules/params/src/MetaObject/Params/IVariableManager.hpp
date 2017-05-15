#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

#include <vector>
#include <memory>

namespace mo {
class IParam;
template<typename T> class ITParam;
template<typename T> class TInputParam;
class MO_EXPORTS IVariableManager {
public:
    virtual void addParam(IParam* param) = 0;

    virtual void RemoveParam(IParam* param) = 0;

    template<typename T> std::vector<IParam*> getOutputParams();
    virtual std::vector<IParam*> getOutputParams(TypeInfo type) = 0;

    virtual std::vector<IParam*> GetAllParmaeters() = 0;
    virtual std::vector<IParam*> GetAllOutputParams() = 0;


    virtual IParam* getOutputParam(std::string name) = 0;
    virtual IParam* getParam(std::string name) = 0;

    // Links an output Param to an input Param with either a direct link or a buffered link.
    // Buffered links allow Params to desync frame number between the producer and the consumer
    virtual void LinkParams(IParam* output, IParam* input) = 0;
};

template<typename T> std::vector<IParam*> IVariableManager::getOutputParams() {
    return getOutputParams(TypeInfo(typeid(T)));
}
}
