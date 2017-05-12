#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<EnumParam, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), enumCombo(nullptr){}


void THandler<EnumParam, void>::updateUi( const EnumParam& data){
    _updating = true;
    enumCombo->clear();
    for (int i = 0; i < data.enumerations.size(); ++i){
        enumCombo->addItem(QString::fromStdString(data.enumerations[i]));
    }
    enumCombo->setCurrentIndex(data.currentSelection);
    _updating = false; 
}

void THandler<EnumParam, void>::updateParam(EnumParam& data){
    if(enumCombo){
        data.currentSelection = enumCombo->currentIndex();
    }
}

std::vector<QWidget*> THandler<EnumParam, void>::getUiWidgets(QWidget* parent){

    std::vector<QWidget*> output;
    if (enumCombo == nullptr)
        enumCombo = new QComboBox(parent);
    enumCombo->connect(enumCombo, SIGNAL(currentIndexChanged(int)), proxy, SLOT(on_update(int)));
    output.push_back(enumCombo);
    return output;
}
#endif
