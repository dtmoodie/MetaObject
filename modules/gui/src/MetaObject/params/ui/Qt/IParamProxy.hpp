#pragma once

#include "MetaObject/detail/Export.hpp"
#include <memory>
class QWidget;
class QObject;
namespace mo
{
    class IControlParam;
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                IParamProxy
            // *****************************************************************************
            class MO_EXPORTS IParamProxy
            {
              protected:
              public:
                typedef std::shared_ptr<IParamProxy> Ptr;
                IParamProxy();
                virtual ~IParamProxy();

                virtual QWidget* getParamWidget(QWidget* parent) = 0;
                virtual bool checkParam(IControlParam* param) const = 0;
                virtual bool setParam(IControlParam* param) = 0;
                virtual void onUiUpdate(QObject* source) = 0;
            };
        } // namespace qt
    }     // namespace UI
} // namespace mo