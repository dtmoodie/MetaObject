#pragma once
#ifdef HAVE_QT5
#include "IHandler.hpp"
#include "IParamProxy.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Params/Demangle.hpp"

class QWidget;
class QObject;
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                Unspecialized handler
            // *****************************************************************************
            template<typename T, typename Enable = void>
            class THandler : public IHandler{
                IParamProxy& _parent;
            public:
                THandler(IParamProxy& parent):
                    _parent(parent){
                    LOG(debug) << "Creating handler for default unspecialized Param " << Demangle::TypeToName(typeid(T));
                }

                // Update user interface from parameter update
                void updateUi(const T& data){}

                // update raw data from user interface
                void updateParam(T& data){}

                // notify parent ParamProxy of the update, param proxy will lock parameter and call updateParam
                void onUiUpdate(QObject* sender){ _parent.onUiUpdate(sender); }
                
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    LOG(debug) << "Creating widget for default unspecialized Param " << Demangle::TypeToName(typeid(T));
                    return std::vector<QWidget*>();
                }
                
                static const bool UI_UPDATE_REQUIRED = false;
                static const bool IS_DEFAULT = true;
            };
        }
    }
}
#endif // HAVE_QT5
