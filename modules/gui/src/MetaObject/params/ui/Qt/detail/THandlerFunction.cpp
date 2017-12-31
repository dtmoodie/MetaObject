#ifdef HAVE_QT5
#include "MetaObject/params/Types.hpp"
#include "MetaObject/params/ui/Qt/POD.hpp"
#include "qpushbutton.h"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::function<void(void)>, void>::THandler(IParamProxy& parent) : UiUpdateHandler(parent), btn(nullptr)
{
}

void THandler<std::function<void(void)>, void>::updateUi(const std::function<void(void)>& data)
{
    (void)data;
}

void THandler<std::function<void(void)>, void>::updateParam(std::function<void(void)>& data)
{
    data();
}

std::vector<QWidget*> THandler<std::function<void(void)>, void>::getUiWidgets(QWidget* parent)
{
    std::vector<QWidget*> output;
    if (btn == nullptr)
    {
        btn = new QPushButton(parent);
    }
    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

#endif
