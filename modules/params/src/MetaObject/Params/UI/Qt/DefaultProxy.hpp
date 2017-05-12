#pragma once
#include "IParamProxy.hpp"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Signals/TSlot.hpp"
#include "MetaObject/Params/IParam.hpp"
namespace mo
{
    class Context;
    class IParam;
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                DefaultProxy
            // *****************************************************************************
            class MO_EXPORTS DefaultProxy: public IParamProxy
            {
            public:
                DefaultProxy(IParam* param);
                virtual bool checkParam(IParam* param) const;
                bool setParam(IParam* param);
                QWidget* getParamWidget(QWidget* parent);
            protected:
                IParam* param;
                TSlot<void(IParam const*)> delete_slot;
                UpdateSlot_t update_slot;
                virtual void onUiUpdate(QObject* source);
                virtual void onParamUpdate(IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
                virtual void onParamDelete(IParam const*);
            };
        }
    }
}