
#include "MetaObject/Params/UI/Qt/DefaultProxy.hpp"
#include "MetaObject/Params/IParam.hpp"

using namespace mo;
using namespace mo::UI::qt;

DefaultProxy::DefaultProxy(IParam* param)
{
    delete_slot = std::bind(&DefaultProxy::onParamDelete, this, std::placeholders::_1);
    update_slot = std::bind(&DefaultProxy::onParamUpdate, this, std::placeholders::_1, std::placeholders::_2);
    Param = param;
    param->registerUpdateNotifier(&update_slot);
    param->registerDeleteNotifier(&delete_slot);
}
bool DefaultProxy::SetParam(IParam* param)
{
    Param = param;
    param->registerUpdateNotifier(&update_slot);
    param->registerDeleteNotifier(&delete_slot);
    return true;
}
bool DefaultProxy::CheckParam(IParam* param)
{    
    return param == Param;
}
#ifdef HAVE_QT5
#include <qgridlayout.h>
#include <qlabel.h>
#include <qstring.h>
#endif
QWidget* DefaultProxy::getParamWidget(QWidget* parent)
{    
#ifdef HAVE_QT5
    QWidget* output = new QWidget(parent);

    QGridLayout* layout = new QGridLayout(output);
    QLabel* nameLbl = new QLabel(QString::fromStdString(Param->getName()), output);
    nameLbl->setToolTip(QString::fromStdString(Param->getTypeInfo().name()));
    layout->addWidget(nameLbl, 0, 0);
    output->setLayout(layout);
    return output;
#endif
	return nullptr;
}

void DefaultProxy::onUiUpdate()
{
}

void DefaultProxy::onParamUpdate(Context* ctx, IParam* param)
{

}

void DefaultProxy::onParamDelete(IParam const* param)
{
    Param = nullptr;
}