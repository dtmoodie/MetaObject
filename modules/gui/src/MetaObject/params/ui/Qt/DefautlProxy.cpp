#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ui/Qt/DefaultProxy.hpp"
#include <qgridlayout.h>
#include <qlabel.h>
#include <qstring.h>

using namespace mo;
using namespace mo::UI::qt;

DefaultProxy::DefaultProxy(IParam* param)
{
    delete_slot = std::bind(&DefaultProxy::onParamDelete, this, std::placeholders::_1);
    update_slot = std::bind(
        &DefaultProxy::onParamUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    this->param = param;
    param->registerUpdateNotifier(&update_slot);
    param->registerDeleteNotifier(&delete_slot);
}
bool DefaultProxy::setParam(IParam* param)
{
    this->param = param;
    param->registerUpdateNotifier(&update_slot);
    param->registerDeleteNotifier(&delete_slot);
    return true;
}
bool DefaultProxy::checkParam(IParam* param) const
{
    return this->param == param;
}

QWidget* DefaultProxy::getParamWidget(QWidget* parent)
{
    QWidget* output = new QWidget(parent);

    QGridLayout* layout = new QGridLayout(output);
    QLabel* nameLbl = new QLabel(QString::fromStdString(param->getName()), output);
    nameLbl->setToolTip(QString::fromStdString(param->getTypeInfo().name()));
    layout->addWidget(nameLbl, 0, 0);
    output->setLayout(layout);
    return output;
}

void DefaultProxy::onUiUpdate(QObject* source)
{
    (void)source;
}

void DefaultProxy::onParamUpdate(IParam*, Header, UpdateFlags)
{
}

void DefaultProxy::onParamDelete(IParam const* param)
{
    param = nullptr;
}
