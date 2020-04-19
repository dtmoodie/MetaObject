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
    class IPublisher;
    class ISubscriber;

    class MO_EXPORTS IParamServer
    {
      public:
        virtual ~IParamServer();
        virtual void addParam(IMetaObject& obj, IParam& param) = 0;

        virtual void removeParam(const IParam& param) = 0;
        virtual void removeParams(const IMetaObject& obj) = 0;

        template <typename T>
        std::vector<IPublisher*> getPublishers();
        virtual std::vector<IPublisher*> getPublishers(TypeInfo type = TypeInfo::Void()) = 0;

        virtual std::vector<IParam*> getAllParms() = 0;

        virtual IPublisher* getPublisher(std::string name) = 0;
        virtual IParam* getParam(std::string name) = 0;

        // Links an output Param to an input Param with either a direct link or a buffered link.
        // Buffered links allow Params to desync frame number between the producer and the consumer
        virtual void linkParams(IPublisher& publisher, ISubscriber& subscriber) = 0;
    };

    template <typename T>
    std::vector<IPublisher*> IParamServer::getPublishers()
    {
        return getPublishers(TypeInfo::create<T>());
    }
} // namespace mo
