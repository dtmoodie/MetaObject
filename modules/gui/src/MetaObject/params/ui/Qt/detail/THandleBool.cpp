#ifdef HAVE_QT5
#include "MetaObject/params/ui/Qt/POD.hpp"
#include "MetaObject/params/ui/Qt/SignalProxy.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qcheckbox.h"

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<bool,void>::THandler(IParamProxy& parent):
    UiUpdateHandler(parent), chkBox(nullptr){
}

void THandler<bool,void>::updateUi( const bool& data){
    _updating= true;
    if(chkBox)
        chkBox->setChecked(data);
    _updating = false;
}


void THandler<bool, void>::updateParam(bool& data){
    if(chkBox)
        data = chkBox->isChecked();
}

std::vector < QWidget*> THandler<bool,void>::getUiWidgets(QWidget* parent_){
    std::vector<QWidget*> output;
    _parent_widget = parent_;
    if (chkBox == nullptr)
        chkBox = new QCheckBox(parent_);
    chkBox->connect(chkBox, SIGNAL(stateChanged(int)), IHandler::proxy, SLOT(on_update(int)));
    output.push_back(chkBox);
    return output;
}

#endif
