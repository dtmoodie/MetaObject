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
        void addParam(IMetaObject* obj, IParam* param) override;
        void removeParam(IMetaObject* obj, IParam* param) override;
        void removeParam(const IMetaObject* obj) override;

        std::vector<IParam*> getOutputParams(TypeInfo type) override;
        std::vector<IParam*> getAllParms() override;
        std::vector<IParam*> getAllOutputParams() override;

        IParam* getOutputParam(std::string name) override;
        IParam* getParam(std::string name) override;
        void linkParams(IParam* output, IParam* input) override;

      private:
        std::map<std::string, IParam*> _params;
        TSlot<void(IMetaObject*, IParam*)> delete_slot;
        std::map<const IMetaObject*, std::vector<std::string>> _obj_params;
    };
}

#endif // MO_PARAMS_PARAM_SERVER_HPP
