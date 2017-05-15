#pragma once
#ifdef HAVE_QT5
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/core/detail/Forward.hpp"
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
            public:
                static const bool UI_UPDATE_REQUIRED = false;
                IHandler();
                virtual ~IHandler();
                virtual void onUiUpdate(QObject* sender);
                virtual void onUiUpdate(QObject* sender, double val);
                virtual void onUiUpdate(QObject* sender, int val);
                virtual void onUiUpdate(QObject* sender, bool val);
                virtual void onUiUpdate(QObject* sender, QString val);
                virtual void onUiUpdate(QObject* sender, int row, int col);
                
                virtual std::function<void(void)>& getOnUpdate();
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent);
            protected:
                SignalProxy* proxy;
                std::function<void(void)> onUpdate;
            };
        }
    }
}
#endif
