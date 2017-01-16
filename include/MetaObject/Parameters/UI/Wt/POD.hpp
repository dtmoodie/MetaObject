#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>
#include <Wt/WSpinBox>
namespace mo
{
    namespace UI
    {
        namespace wt
        {
            template<class T>
            class TParameterProxy<T, typename std::enable_if<std::is_integral<T>::value>::type> : public IParameterProxy
            {
            public:
                static const int IS_DEFAULT = false;
                TParameterProxy(ITypedParameter<T>* param_, MainApplication* app_,
                    WContainerWidget *parent = 0) :
                    IParameterProxy(param_, app_, parent),
                    _param(param_)
                {
                    _spin_box = new Wt::WSpinBox(this);
                    _spin_box->setValue(param_->GetData());
                }
            protected:
                void SetTooltip(const std::string& tip)
                {
                    auto lock = _app->getUpdateLock();
                    _spin_box->setToolTip(tip);
                    _app->requestUpdate();
                }
                void onUpdate(mo::Context* ctx, mo::IParameter* param)
                {
                    auto lock = _app->getUpdateLock();
                    _spin_box->setValue(_param->GetData());
                    _app->requestUpdate();
                }
                mo::ITypedParameter<T>* _param;
                Wt::WSpinBox* _spin_box;
            };
        } // namespace wt
    } // namespace UI
} // namespace mo
