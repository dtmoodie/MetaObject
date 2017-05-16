#ifdef HAVE_WT
#include <MetaObject/params/ui/Wt/IParamProxy.hpp>
#include <MetaObject/params/IParam.hpp>

using namespace mo::UI::wt;
using namespace mo;

IParamProxy::IParamProxy(IParam* param_, MainApplication* app_,
    WContainerWidget *parent_) :
    Wt::WContainerWidget(parent_),
    _app(app_),
    _onUpdateSlot(std::bind(&IParamProxy::onParamUpdate, this,
        std::placeholders::_1, std::placeholders::_2))
{
    _onUpdateConnection = param_->registerUpdateNotifier(&_onUpdateSlot);
    auto text = new Wt::WText(param_->getTreeName(), this);
    text->setToolTip(Demangle::TypeToName(param_->getTypeInfo()));
    this->addWidget(text);
}

IParamProxy::~IParamProxy()
{

}

IPlotProxy::IPlotProxy(IParam *param_, MainApplication *app_, WContainerWidget *parent_):
    Wt::WContainerWidget(parent_),
    _app(app_),
    _onUpdateSlot(std::bind(&IPlotProxy::onParamUpdate, this,
        std::placeholders::_1, std::placeholders::_2))
{
    _onUpdateConnection = param_->registerUpdateNotifier(&_onUpdateSlot);
}
IPlotProxy::~IPlotProxy()
{

}

#endif
