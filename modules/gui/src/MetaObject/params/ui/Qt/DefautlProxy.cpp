#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/ui/Qt/DefaultProxy.hpp"
#include <qgridlayout.h>
#include <qlabel.h>
#include <qstring.h>

using namespace mo;
using namespace mo::UI::qt;

DefaultProxy::DefaultProxy(IControlParam* param)
{
    delete_slot.bind(&DefaultProxy::onParamDelete, this);
    update_slot.bind(&DefaultProxy::onParamUpdate, this);
    this->m_param = param;
    m_param->registerUpdateNotifier(update_slot);
    m_param->registerDeleteNotifier(delete_slot);
}
bool DefaultProxy::setParam(IControlParam* param)
{
    this->m_param = param;
    m_param->registerUpdateNotifier(update_slot);
    m_param->registerDeleteNotifier(delete_slot);
    return true;
}
bool DefaultProxy::checkParam(IControlParam* param) const
{
    return this->m_param == param;
}

QWidget* DefaultProxy::getParamWidget(QWidget* parent)
{
    QWidget* output = new QWidget(parent);

    QGridLayout* layout = new QGridLayout(output);
    QLabel* nameLbl = new QLabel(QString::fromStdString(m_param->getName()), output);
    nameLbl->setToolTip(QString::fromStdString(m_param->getTypeInfo().name()));
    layout->addWidget(nameLbl, 0, 0);
    output->setLayout(layout);
    return output;
}

void DefaultProxy::onUiUpdate(QObject* source)
{
    (void)source;
}

void DefaultProxy::onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream&)
{
}

void DefaultProxy::onParamDelete(const IParam& param)
{
    m_param = nullptr;
}
