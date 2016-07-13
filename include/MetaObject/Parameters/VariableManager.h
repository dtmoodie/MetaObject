#pragma once

#include "MetaObject/Parameters/IVariableManager.h"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Detail/Export.hpp"

namespace mo
{
    class IParameter;
    class MO_EXPORTS VariableManager: public IVariableManager
    {
    public:
        VariableManager();
        ~VariableManager();
        virtual void AddParameter(std::shared_ptr<IParameter> param);
        virtual void RemoveParameter(IParameter* param);

        virtual std::vector<std::shared_ptr<IParameter>> GetOutputParameters(TypeInfo type);
        virtual std::vector<std::shared_ptr<IParameter>> GetAllParmaeters();
        virtual std::vector<std::shared_ptr<IParameter>> GetAllOutputParameters();

        virtual std::shared_ptr<IParameter> GetOutputParameter(std::string name);
        virtual std::shared_ptr<IParameter> GetParameter(std::string name);
        virtual void LinkParameters(std::weak_ptr<IParameter> output, std::weak_ptr<IParameter> input);

    private:
        struct impl;
        impl* pimpl;
    };

    // By default uses a buffered connection when linking two parameters together
    class MO_EXPORTS BufferedVariableManager : public VariableManager
    {
    };
}
