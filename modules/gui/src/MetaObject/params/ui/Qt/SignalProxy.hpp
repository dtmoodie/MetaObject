#pragma once
#ifdef HAVE_QT5

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#ifdef metaobject_qtgui_EXPORTS
#  define MO_QTGUI_EXPORTS __declspec(dllexport)
#else
#  define MO_QTGUI_EXPORTS __declspec(dllimport)
#endif
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_QTGUI_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_QTGUI_EXPORTS
#endif

#include <qobject.h>
#include <qdatetime.h>

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            class IHandler;
            // *****************************************************************************
            //                                SignalProxy
            // *****************************************************************************
            class MO_QTGUI_EXPORTS SignalProxy : public QObject
            {
                Q_OBJECT
                IHandler* handler;
                QTime lastCallTime;
            public:
                SignalProxy(IHandler* handler_);

            public slots:
                void on_update();
                void on_update(int);
                void on_update(double);
                void on_update(bool);
                void on_update(QString);
                void on_update(int row, int col);
            };
        }
    }
}
#endif