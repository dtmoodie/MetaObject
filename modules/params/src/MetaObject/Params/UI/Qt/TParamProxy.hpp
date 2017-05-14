#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/TypeInfo.hpp"
#include "MetaObject/Params/UI/WidgetFactory.hpp"
#include "MetaObject/Params/ITAccessibleParam.hpp"
#include "IParamProxy.hpp"
#include "THandler.hpp"
#include "qwidget.h"
#include "qgridlayout.h"
#include "qpushbutton.h"
#include "qlabel.h"
class QWidget;

namespace mo{
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
    template<typename T> class ParamProxy : public IParamProxy{
    public:
        ParamProxy(IParam* param):
            _param_handler(*this){
            setParam(param);
        }
        ~ParamProxy(){
        
        }
                
        QWidget* getParamWidget(QWidget* parent){
            QWidget* output = new QWidget(parent);
            auto widgets = _param_handler.getUiWidgets(output);
            QGridLayout* layout = new QGridLayout(output);
            if (_param->getTypeInfo() == TypeInfo(typeid(std::function<void(void)>))){
                dynamic_cast<QPushButton*>(widgets[0])->setText(QString::fromStdString(_param->getName()));
                layout->addWidget(widgets[0], 0, 0);
            }else{
                QLabel* name_lbl = new QLabel(QString::fromStdString(_param->getName()), output);
                name_lbl->setToolTip(QString::fromStdString(_param->getTypeInfo().name()));
                layout->addWidget(name_lbl, 0, 0);
                int count = 1;
                output->setLayout(layout);
                for (auto itr = widgets.rbegin(); itr != widgets.rend(); ++itr, ++count){
                    layout->addWidget(*itr, 0, count);
                }
                // Correct the tab order of the widgets
                for (size_t i = widgets.size() - 1; i > 0; --i){
                    QWidget::setTabOrder(widgets[i], widgets[i - 1]);
                }
                auto token = _param->access();
                _param_handler.setUpdating(true);
                _param_handler.updateUi(token());
                _param_handler.setUpdating(false);
            }
            return output;
        }

        bool checkParam(IParam* param) const{
            return param == this->_param;
        }

        bool setParam(IParam* param){
            this->_param = dynamic_cast<ITAccessibleParam<T>*>(param);
            if(this->_param){
                auto token = this->_param->access();
                _param_handler.setUpdating(true);
                _param_handler.updateUi(token());
                _param_handler.setUpdating(false);
                return true;
            }
            return false;
        }
    protected:
        void onParamUpdate(typename ParamTraits<T>::ConstStorageRef_t data, IParam* param, Context* ctx, OptionalTime_t ts, size_t fn, ICoordinateSystem* cs, UpdateFlags fg){
            if(param == this->param){
                param_handler.updateUi(ParamTraits<T>::get(data));
            }
        }

        void onParamDelete(IParam const* param){}

        virtual void onUiUpdate(QObject* source){
            auto token = _param->access();
            _param_handler.setUpdating(true);
            _param_handler.updateParam(token());
            _param_handler.setUpdating(false);
        }
        THandler<T> _param_handler;
        ITAccessibleParam<T>* _param;
        typename ITParam<T>::TUpdateSlot_t _slot;
    };
    // **********************************************************************************
    // *************************** Constructor ******************************************
    // **********************************************************************************

    template<typename T> class Constructor{
    public:
        Constructor(){
            if(!ParamProxy<T>::IS_DEFAULT)
                WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)), std::bind(&Constructor<T>::Create, std::placeholders::_1));
        }
        static std::shared_ptr<IParamProxy> Create(IParam* param){
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
//#include "detail/TParamProxyImpl.hpp"
#endif // HAVE_QT5
