#pragma once

#include "MetaObject/Detail/Export.hpp"
#include <memory>
class QWidget;

namespace mo
{
    class IParam;
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
                virtual bool checkParam(IParam* param) = 0;
                virtual bool setParam(IParam* param) = 0;
            };
        }
    }
}