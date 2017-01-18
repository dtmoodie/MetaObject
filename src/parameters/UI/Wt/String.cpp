#ifdef HAVE_WT
#include <MetaObject/Parameters/UI/Wt/String.hpp>
using namespace mo::UI::wt;
using namespace mo;



TParameterProxy<std::string, void>::TParameterProxy(ITypedParameter<std::string>* param_, MainApplication* app_,
    WContainerWidget *parent_) :
    IParameterProxy(param_, app_, parent_),
    _param(param_)
{
    _line_edit = new Wt::WLineEdit(this);
    _line_edit->setText(param_->GetData());
}

void TParameterProxy<std::string, void>::SetTooltip(const std::string& tip)
{
    auto lock = _app->getUpdateLock();
    _line_edit->setToolTip(tip);
    _app->requestUpdate();
}

void TParameterProxy<std::string, void>::onUpdate(mo::Context* ctx, mo::IParameter* param)
{
    auto lock = _app->getUpdateLock();
    _line_edit->setText(_param->GetData());
    _app->requestUpdate();
}
#endif
