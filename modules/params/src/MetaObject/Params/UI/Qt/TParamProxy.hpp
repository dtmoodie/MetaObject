#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/TypeInfo.hpp"
#include "MetaObject/Params/UI/WidgetFactory.hpp"
#include "IParamProxy.hpp"
#include "THandler.hpp"
class QWidget;

namespace mo
{
    class IParam;
    class Context;
    template<typename T> class ITParam;
    template<typename T> class ITRangedParam;
    template<typename T> class THandler;
    template<typename T, int N, typename Enable> struct MetaParam;
    namespace UI
    {
        namespace qt
        {
            // **********************************************************************************
            // *************************** ParamProxy ***************************************
            // **********************************************************************************  
            template<typename T> class ParamProxy : public IParamProxy
            {
            public:
                static const bool IS_DEFAULT = THandler<T>::IS_DEFAULT;

                ParamProxy(IParam* param);
                ~ParamProxy();
                
                QWidget* getParamWidget(QWidget* parent);
                bool checkParam(IParam* param);
                bool setParam(IParam* param);
            protected:
                void onParamUpdate(Context* ctx, IParam* param);
                void onParamDelete(IParam const* param);
                void onUiUpdate();
                THandler<T> paramHandler;
                ITParam<T>* Param;
            };
            // **********************************************************************************
            // *************************** Constructor ******************************************
            // **********************************************************************************

            template<typename T> class Constructor
            {
            public:
                Constructor()
                {
                    if(!ParamProxy<T>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)), std::bind(&Constructor<T>::Create, std::placeholders::_1));
                }
                static std::shared_ptr<IParamProxy> Create(IParam* param)
                {
                    return std::shared_ptr<IParamProxy>(new ParamProxy<T>(param));
                }
            };
        }
    }
#define MO_UI_QT_PARAMTERPROXY_METAParam(N) \
            template<class T> struct MetaParam<T, N, typename std::enable_if<!UI::qt::ParamProxy<T>::IS_DEFAULT, void>::type>: public MetaParam<T, N-1, void> \
            { \
                static UI::qt::Constructor<T> _Param_proxy_constructor; \
                MetaParam(const char* name): \
                    MetaParam<T, N-1, void>(name) \
                { \
                    (void)&_Param_proxy_constructor; \
                } \
            }; \
            template<class T> UI::qt::Constructor<T> MetaParam<T,N, typename std::enable_if<!UI::qt::ParamProxy<T>::IS_DEFAULT, void>::type>::_Param_proxy_constructor;

    MO_UI_QT_PARAMTERPROXY_METAParam(__COUNTER__)
}
#include "detail/TParamProxyImpl.hpp"
#endif // HAVE_QT5
