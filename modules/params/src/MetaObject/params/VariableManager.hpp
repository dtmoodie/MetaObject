#pragma once

#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/IVariableManager.hpp"
#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/signals/TSlot.hpp"

#include <map>
namespace mo
{
    class IParam;
    class MO_EXPORTS VariableManager : public IVariableManager
    {
      public:
        VariableManager();
        virtual ~VariableManager();
        virtual void addParam(IMetaObject* obj, IParam* param);
        virtual void removeParam(IMetaObject* obj, IParam* param);
        virtual void removeParam(const IMetaObject* obj);

        virtual std::vector<IParam*> getOutputParams(TypeInfo type);
        virtual std::vector<IParam*> getAllParms();
        virtual std::vector<IParam*> getAllOutputParams();

        virtual IParam* getOutputParam(std::string name);
        virtual IParam* getParam(std::string name);
        virtual void linkParams(IParam* output, IParam* input);

      private:
        std::map<std::string, IParam*> _params;
        TSlot<void(IMetaObject*, IParam*)> delete_slot;
        std::map<const IMetaObject*, std::vector<std::string>> _obj_params;
    };

    // By default uses a buffered Connection when linking two Params together
    class MO_EXPORTS BufferedVariableManager : public VariableManager
    {
    };
}
