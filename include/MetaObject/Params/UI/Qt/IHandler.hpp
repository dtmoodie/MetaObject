#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Forward.hpp"
#include <qstring.h>
#include <functional>
#include <vector>

class QObject;
class QWidget;

namespace mo{
    namespace UI{
        namespace qt{
            class SignalProxy;
            class IHandler;
            struct MO_EXPORTS UiUpdateListener{
                virtual void onUpdate(IHandler* handler) = 0;
            };
            // *****************************************************************************
            //                                IHandler
            // *****************************************************************************
            class MO_EXPORTS IHandler{
            private:
                Mutex_t** paramMtx;
            protected:
                UiUpdateListener* _listener;
                SignalProxy* proxy;
                std::function<void(void)> onUpdate;
            public:
                IHandler();
                virtual ~IHandler();
                virtual void onUiUpdate(QObject* sender);
                virtual void onUiUpdate(QObject* sender, double val);
                virtual void onUiUpdate(QObject* sender, int val);
                virtual void onUiUpdate(QObject* sender, bool val);
                virtual void onUiUpdate(QObject* sender, QString val);
                virtual void onUiUpdate(QObject* sender, int row, int col);
                virtual void setUpdateListener(UiUpdateListener* listener);
                
                virtual std::function<void(void)>& getOnUpdate();
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent);
                virtual void setParamMtx(Mutex_t** mtx);
                virtual Mutex_t* getParamMtx();
                static bool uiUpdateRequired();
            };
        }
    }
}
#endif
