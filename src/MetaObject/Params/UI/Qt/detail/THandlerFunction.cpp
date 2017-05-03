#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/POD.hpp"
#include "MetaObject/Params/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qpushbutton.h"
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::function<void(void)>, void>::THandler() : 
    funcData(nullptr), 
    btn(nullptr) 
{}

void THandler<std::function<void(void)>, void>::UpdateUi(std::function<void(void)>* data)
{
    funcData = data;
}

void THandler<std::function<void(void)>, void>::onUiUpdate(QObject* sender)
{
    if (sender == btn && IHandler::getParamMtx())
    {
        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
        if (funcData)
        {
            (*funcData)();
            if (onUpdate)
            {
                onUpdate();
                if(_listener)
                    _listener->onUpdate(this);
            }
        }
    }
}

void THandler<std::function<void(void)>, void>::SetData(std::function<void(void)>* data_)
{
    funcData = data_;
}

std::function<void(void)>* THandler<std::function<void(void)>, void>::GetData()
{
    return funcData;
}

std::vector<QWidget*> THandler<std::function<void(void)>, void>::GetUiWidgets(QWidget* parent)
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