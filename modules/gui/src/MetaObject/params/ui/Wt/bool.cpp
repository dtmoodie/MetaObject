#ifdef HAVE_WT
#include "MetaObject/params/UI/Wt/POD.hpp"
#include <Wt/WCheckBox>

using namespace mo::UI::wt;

TDataProxy<bool, void>::TDataProxy():
    _check_box(nullptr)
{

}

void TDataProxy<bool, void>::CreateUi(IParamProxy* proxy, bool* data, bool read_only)
{
    _check_box = new Wt::WCheckBox(proxy);
    _check_box->changed().connect(proxy, &IParamProxy::onUiUpdate);
    _check_box->setReadOnly(read_only);
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
#endif
