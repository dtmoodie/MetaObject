#include <MetaObject/Parameters/UI/Wt/String.hpp>
using namespace mo::UI::wt;
using namespace mo;



TDataProxy<std::string, void>::TDataProxy(IParameterProxy& proxy):
    _proxy(proxy)
{

}

void TDataProxy<std::string, void>::SetTooltip(const std::string& tp)
{

}

void TDataProxy<std::string, void>::CreateUi(IParameterProxy* proxy, std::string* data)
{
    if(_line_edit)
    {
        delete _line_edit;
        _line_edit = nullptr;
    }
    _line_edit = new Wt::WLineEdit(proxy);
    if(data)
    {
        _line_edit->setText(*data);
    }
    _line_edit->enterPressed().connect(proxy, &IParameterProxy::onUiUpdate);
}

void TDataProxy<std::string, void>::UpdateUi(const std::string& data)
{
    _line_edit->setText(data);
}

void TDataProxy<std::string, void>::onUiUpdate(std::string& data)
{
    data = _line_edit->text().toUTF8();
}
