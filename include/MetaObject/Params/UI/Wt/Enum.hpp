#pragma once

#include "IParamProxy.hpp"
#include <MetaObject/Params/ITParam.hpp>
#include <MetaObject/Params/Types.hpp>

#include <Wt/WComboBox>

namespace mo
{
namespace UI
{
namespace wt
{

    template<>
    class TParamProxy<EnumParam, void> : public IParamProxy
    {
    public:
        TParamProxy(ITParam<EnumParam>* param_,
            MainApplication* app_,
            WContainerWidget* parent_ = 0);
    protected:
        void SetTooltip(const std::string& tip);
        void onParamUpdate(mo::Context* ctx, mo::IParam* param);
        void onUiChanged();

        ITParam<EnumParam>* _param;
        Wt::WComboBox* _combo_box;
    };
}
}
}
