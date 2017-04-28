#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qlineedit.h"

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::string, void>::THandler() : 
    strData(nullptr), 
    lineEdit(nullptr) 
{
}

void THandler<std::string, void>::UpdateUi( std::string* data)
{
    if(data)
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _currently_updating = true;
        lineEdit->setText(QString::fromStdString(*data));
        _currently_updating = false;
    }                    
}
void THandler<std::string, void>::onUiUpdate(QObject* sender)
{
    if(_currently_updating || !IHandler::getParamMtx())
        return;
    if (sender == lineEdit && strData)
    {    
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        *strData = lineEdit->text().toStdString();
        if (onUpdate)
            onUpdate();
        if(_listener)
            _listener->OnUpdate(this);
    }
}
void THandler<std::string, void>::SetData(std::string* data_)
{
    strData = data_;
    if (lineEdit)
    {
        _currently_updating = true;
        lineEdit->setText(QString::fromStdString(*strData));
        _currently_updating = false;
    }

}
std::string* THandler<std::string, void>::GetData()
{
    return strData;
}

std::vector<QWidget*> THandler<std::string, void>::GetUiWidgets(QWidget* parent)
{

    std::vector<QWidget*> output;
    if (lineEdit == nullptr)
        lineEdit = new QLineEdit(parent);
    //lineEdit->connect(lineEdit, SIGNAL(editingFinished()), proxy, SLOT(on_update()));
    lineEdit->connect(lineEdit, SIGNAL(returnPressed()), proxy, SLOT(on_update()));
    output.push_back(lineEdit);
    return output;
}
#endif