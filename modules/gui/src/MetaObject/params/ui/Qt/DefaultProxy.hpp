#pragma once
#include "IParamProxy.hpp"
#include <MetaObject/params/IControlParam.hpp>
#include <MetaObject/signals/Connection.hpp>
#include <MetaObject/signals/TSlot.hpp>
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
            class MO_EXPORTS DefaultProxy : public IParamProxy
            {
              public:
                DefaultProxy(IControlParam* param);
                virtual bool checkParam(IControlParam* param) const;
                bool setParam(IControlParam* param);
                QWidget* getParamWidget(QWidget* parent);

              protected:
                IControlParam* m_param;
                TSlot<void(const IParam&)> delete_slot;
                UpdateSlot_t update_slot;
                virtual void onUiUpdate(QObject* source);
                virtual void onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream&);
                virtual void onParamDelete(const IParam&);
            };
        } // namespace qt
    }     // namespace UI
} // namespace mo
