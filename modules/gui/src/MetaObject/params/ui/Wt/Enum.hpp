#pragma once

#include "IParamProxy.hpp"
#include <MetaObject/params/ITParam.hpp>
#include <MetaObject/types/file_types.hpp>

#include <Wt/WComboBox>

namespace mo
{
    namespace UI
    {
        namespace wt
        {

            template <>
            class TParamProxy<EnumParam, void> : public IParamProxy
            {
              public:
                TParamProxy(ITAccessibleParam<EnumParam>* param_, MainApplication* app_, WContainerWidget* parent_ = 0);

              protected:
                void SetTooltip(const std::string& tip);
                void onParamUpdate(mo::IAsyncStream* ctx, mo::IParam* param);
                void onUiChanged();

                ITAccessibleParam<EnumParam>* _param;
                Wt::WComboBox* _combo_box;
            };
        }
    }
}
