#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <MetaObject/core/detail/forward.hpp>
#include <memory>
#include <vector>

namespace mo
{
    template <typename T>
    class TInputParam;
    class IMetaObject;

    class MO_EXPORTS IParamServer
    {
      public:
        virtual ~IParamServer();
        virtual void addParam(IMetaObject* obj, IParam* param) = 0;

        virtual void removeParam(IMetaObject* obj, IParam* param) = 0;
        virtual void removeParam(const IMetaObject* obj) = 0;

        template <typename T>
        std::vector<IParam*> getOutputParams();
        virtual std::vector<IParam*> getOutputParams(TypeInfo type) = 0;

        virtual std::vector<IParam*> getAllParms() = 0;
        virtual std::vector<IParam*> getAllOutputParams() = 0;

        virtual IParam* getOutputParam(std::string name) = 0;
        virtual IParam* getParam(std::string name) = 0;

        // Links an output Param to an input Param with either a direct link or a buffered link.
        // Buffered links allow Params to desync frame number between the producer and the consumer
        virtual void linkParams(IParam* output, IParam* input) = 0;
    };

    template <typename T>
    std::vector<IParam*> IParamServer::getOutputParams()
    {
        return getOutputParams(TypeInfo(typeid(T)));
    }
}
