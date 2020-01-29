#ifndef MO_PARAMS_TPARAMOUTPUT_HPP
#define MO_PARAMS_TPARAMOUTPUT_HPP

#include <MetaObject/params/OutputParam.hpp>
#include <MetaObject/params/TParam.hpp>

namespace mo
{

    template <typename T, uint64_t FLAGS = ParamFlags::kOUTPUT>
    struct MO_EXPORTS TParamOutput : virtual public TParam<T>, virtual public OutputParam
    {
        TParamOutput()
        {
            setFlags(ParamFlags(FLAGS));
        }

        std::vector<TypeInfo> listOutputTypes() const override
        {
            return {TypeInfo(typeid(T))};
        }

        ParamBase* getOutputParam(const TypeInfo) override
        {
            return this;
        }

        const ParamBase* getOutputParam(const TypeInfo) const override
        {
            return this;
        }

        ParamBase* getOutputParam() override
        {
            return this;
        }

        const ParamBase* getOutputParam() const override
        {
            return this;
        }
    };
}
#endif // MO_PARAMS_TPARAMOUTPUT_HPP
