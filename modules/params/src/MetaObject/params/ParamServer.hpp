#ifndef MO_PARAMS_PARAM_SERVER_HPP
#define MO_PARAMS_PARAM_SERVER_HPP

#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/IParamServer.hpp"
#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/signals/TSlot.hpp"

#include <map>
namespace mo
{
    class IParam;

    class MO_EXPORTS ParamServer : public IParamServer
    {
      public:
        ParamServer();
        ~ParamServer() override;
        void addParam(IMetaObject& obj, IParam& param) override;
        void removeParam(const IParam& param) override;
        void removeParams(const IMetaObject& obj) override;

        std::vector<IPublisher*> getPublishers(TypeInfo type = TypeInfo::Void()) override;
        std::vector<IParam*> getAllParms() override;

        IPublisher* getPublisher(std::string name) override;
        IParam* getParam(std::string name) override;
        void linkParams(IPublisher& output, ISubscriber& input) override;

      private:
        std::map<std::string, IParam*> m_params;
        TSlot<void(const IParam&)> m_delete_slot;
        TSlot<void(const IMetaObject&)> m_obj_delete_slot;
        std::map<const IMetaObject*, std::vector<std::string>> m_obj_params;
    };
} // namespace mo

#endif // MO_PARAMS_PARAM_SERVER_HPP
