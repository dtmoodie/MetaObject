#pragma once

#include "MetaObject/params/IVariableManager.hpp"
#include "MetaObject/signals/Connection.hpp"
#include "MetaObject/detail/Export.hpp"

namespace mo {
class IParam;
class MO_EXPORTS VariableManager: public IVariableManager {
public:
    VariableManager();
    ~VariableManager();
    virtual void addParam(IParam* param);
    virtual void removeParam(IParam* param);

    virtual std::vector<IParam*> getOutputParams(TypeInfo type);
    virtual std::vector<IParam*> getAllParms();
    virtual std::vector<IParam*> getAllOutputParams();

    virtual IParam* getOutputParam(std::string name);
    virtual IParam* getParam(std::string name);
    virtual void linkParams(IParam* output, IParam* input);

private:
    struct impl;
    impl* pimpl;
};

// By default uses a buffered Connection when linking two Params together
class MO_EXPORTS BufferedVariableManager : public VariableManager {
};
}