#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<EnumParam, void>::THandler() : enumCombo(nullptr), _updating(false){}
THandler<EnumParam, void>::~THandler()
{

}

void THandler<EnumParam, void>::UpdateUi( EnumParam* data)
{
    if(_updating)
        return;
    if(data)
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        _updating = true;
        enumCombo->clear();
        for (int i = 0; i < data->enumerations.size(); ++i)
        {
            enumCombo->addItem(QString::fromStdString(data->enumerations[i]));
        }
        enumCombo->setCurrentIndex(data->currentSelection);
        _updating = false;
    }                    
}

void THandler<EnumParam, void>::onUiUpdate(QObject* sender, int idx)
{
    if(_updating || !IHandler::getParamMtx())
        return;
    mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
    if (idx != -1 && sender == enumCombo && enumData)
    {
        if(enumData->currentSelection == idx)
            return;
        _updating = true;
        enumData->currentSelection = idx;
        if (onUpdate)
            onUpdate();
        if(_listener)
            _listener->onUpdate(this);
        _updating = false;
    }
}

void THandler<EnumParam, void>::SetData(EnumParam* data_)
{
    enumData = data_;
    if (enumCombo)
        UpdateUi(enumData);
}

EnumParam*  THandler<EnumParam, void>::GetData()
{
    return enumData;
}

std::vector<QWidget*> THandler<EnumParam, void>::GetUiWidgets(QWidget* parent)
{

    std::vector<QWidget*> output;
    if (enumCombo == nullptr)
        enumCombo = new QComboBox(parent);
    enumCombo->connect(enumCombo, SIGNAL(currentIndexChanged(int)), proxy, SLOT(on_update(int)));
    output.push_back(enumCombo);
    return output;
}
#endif
