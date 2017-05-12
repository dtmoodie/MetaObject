#pragma once

#include "MetaObject/Detail/Export.hpp"
#include <memory>
class QWidget;
class QObject;
namespace mo{
class IParam;
namespace UI{ namespace qt{
// *****************************************************************************
//                                IParamProxy
// *****************************************************************************
class MO_EXPORTS IParamProxy{
protected:
public:
    typedef std::shared_ptr<IParamProxy> Ptr;
    IParamProxy();
    virtual ~IParamProxy();
                
    virtual QWidget* getParamWidget(QWidget* parent) = 0;
    virtual bool checkParam(IParam* param) const = 0;
    virtual bool setParam(IParam* param) = 0;
    virtual void onUiUpdate(QObject* source) = 0;
};
} // namespace mo::UI::qt
} // namespace mo::UI
} // namespace mo