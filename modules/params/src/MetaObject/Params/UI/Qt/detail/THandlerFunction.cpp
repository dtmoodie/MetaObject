#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qpushbutton.h"
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::function<void(void)>, void>::THandler(IParamProxy& parent) :
    btn(nullptr),
    UiUpdateHandler(parent)
{}

void THandler<std::function<void(void)>, void>::updateUi(const std::function<void(void)>& data){
    
}

void THandler<std::function<void(void)>, void>::updateParam(std::function<void(void)>& data) {
    data();
}

std::vector<QWidget*> THandler<std::function<void(void)>, void>::getUiWidgets(QWidget* parent){
    std::vector<QWidget*> output;
    if (btn == nullptr){
        btn = new QPushButton(parent);
    }
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

#endif
