#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>
#include <Wt/WLineEdit>

namespace mo
{
    namespace UI
    {
        namespace wt
        {
            template<>
            class MO_EXPORTS TParameterProxy<std::string, void> : public IParameterProxy
            {
            public:
                static const int IS_DEFAULT = false;
                TParameterProxy(ITypedParameter<std::string>* param_, MainApplication* app_,
                    WContainerWidget *parent_ = 0);

            protected:
                void SetTooltip(const std::string& tip);
                void onUpdate(mo::Context* ctx, mo::IParameter* param);
                ITypedParameter<std::string>* _param;
                Wt::WLineEdit* _line_edit;
            };
        } // namespace wt
    } // namespace UI
} // namespace mo
