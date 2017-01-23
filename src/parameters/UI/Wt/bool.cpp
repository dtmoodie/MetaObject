#include "MetaObject/Parameters/UI/Wt/POD.hpp"
#include <Wt/WCheckBox>

using namespace mo::UI::wt;

TDataProxy<bool, void>::TDataProxy(IParameterProxy &proxy):
    _check_box(nullptr),
    _proxy(proxy)
{

}

void TDataProxy<bool, void>::CreateUi(IParameterProxy* proxy, bool* data)
{
    _check_box = new Wt::WCheckBox(proxy);
    _check_box->changed().connect(proxy, &IParameterProxy::onUiUpdate);
    if(data)
    {
        _check_box->setChecked(*data);
    }
}

void TDataProxy<bool, void>::UpdateUi(const bool& data)
{
    _check_box->setChecked(data);
}

void TDataProxy<bool, void>::onUiUpdate(bool& data)
{
    data = _check_box->isChecked();
}

void TDataProxy<bool, void>::SetTooltip(const std::string& tooltip)
{

}
