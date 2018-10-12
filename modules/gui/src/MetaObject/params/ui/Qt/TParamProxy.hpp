#pragma once
#include "IParamProxy.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/ITAccessibleParam.hpp"
#include "MetaObject/params/ui/WidgetFactory.hpp"
#include "THandler.hpp"
#include "qgridlayout.h"
#include "qlabel.h"
#include "qpushbutton.h"
#include "qwidget.h"
class QWidget;

namespace mo
{
    class IParam;
    class Context;
    template <typename T>
    class ITRangedParam;
    template <typename T>
    class THandler;
    template <typename T, int N, typename Enable>
    struct MetaParam;
    namespace UI
    {
        namespace qt
        {
            // **********************************************************************************
            // *************************** ParamProxy ***************************************
            // **********************************************************************************
            template <typename T>
            class ParamProxy : public IParamProxy
            {
              public:
                static const bool IS_DEFAULT = THandler<T>::IS_DEFAULT;
                ParamProxy(IParam* param) : _param_handler(*this) { setParam(param); }
                ~ParamProxy() {}

                QWidget* getParamWidget(QWidget* parent)
                {
                    if (!this->_param)
                        return nullptr;
                    QWidget* output = new QWidget(parent);
                    auto widgets = _param_handler.getUiWidgets(output);
                    QGridLayout* layout = new QGridLayout(output);
                    if (_param->getTypeInfo() == TypeInfo(typeid(std::function<void(void)>)))
                    {
                        dynamic_cast<QPushButton*>(widgets[0])->setText(QString::fromStdString(_param->getName()));
                        layout->addWidget(widgets[0], 0, 0);
                    }
                    else
                    {
                        QLabel* name_lbl = new QLabel(QString::fromStdString(_param->getName()), output);
                        name_lbl->setToolTip(QString::fromStdString(_param->getTypeInfo().name()));
                        layout->addWidget(name_lbl, 0, 0);
                        int count = 1;
                        output->setLayout(layout);
                        for (auto itr = widgets.rbegin(); itr != widgets.rend(); ++itr, ++count)
                        {
                            layout->addWidget(*itr, 0, count);
                        }
                        // Correct the tab order of the widgets
                        for (size_t i = widgets.size() - 1; i > 0; --i)
                        {
                            QWidget::setTabOrder(widgets[i], widgets[i - 1]);
                        }
                        try
                        {
                            auto token = _param->access();
                            _param_handler.setUpdating(true);
                            _param_handler.updateUi(token());
                            _param_handler.setUpdating(false);
                        }
                        catch (mo::ExceptionWithCallStack<std::string>& exc)
                        {
                            (void)exc; // exception thrown if data hasn't been populated yet on some parameters
                        }
                    }
                    return output;
                }

                bool checkParam(IParam* param) const { return param == this->_param; }

                bool setParam(IParam* param)
                {
                    this->_param = dynamic_cast<ITAccessibleParam<T>*>(param);
                    if (this->_param)
                    {
                        return true;
                    }
                    return false;
                }

              protected:
                void onParamUpdate(typename ParamTraits<T>::ConstStorageRef_t data,
                                   IParam* param,
                                   Context* ctx,
                                   OptionalTime ts,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs,
                                   UpdateFlags fg)
                {
                    (void)ctx;
                    (void)ts;
                    (void)fn;
                    (void)cs;
                    (void)fg;
                    if (param == this->param)
                    {
                        // Should handle pushing to UI thread here
                        _param_handler.updateUi(ParamTraits<T>::get(data));
                    }
                }

                void onParamDelete(IParam const* param)
                {
                    if (param == this->_param)
                        this->_param = nullptr;
                }

                virtual void onUiUpdate(QObject* source)
                {
                    (void)source; // should probably redesign to pass into updateParam
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

            template <typename T>
            class Constructor
            {
              public:
                Constructor()
                {
                    if (!ParamProxy<T>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterConstructor(
                            TypeInfo(typeid(T)), std::bind(&Constructor<T>::create, std::placeholders::_1));
                }
                static std::shared_ptr<IParamProxy> create(IParam* param)
                {
                    return std::shared_ptr<IParamProxy>(new ParamProxy<T>(param));
                }
            };
        }
    }
#define MO_UI_QT_PARAMTERPROXY_METAParam(N)                                                                            \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, typename std::enable_if<!UI::qt::ParamProxy<T>::IS_DEFAULT>::type>                          \
        : public MetaParam<T, N - 1, void>                                                                             \
    {                                                                                                                  \
        static UI::qt::Constructor<T> _Param_proxy_constructor;                                                        \
        MetaParam(SystemTable* table, const char* name) : MetaParam<T, N - 1, void>(table, name)                       \
        {                                                                                                              \
            (void)&_Param_proxy_constructor;                                                                           \
        }                                                                                                              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    UI::qt::Constructor<T>                                                                                             \
        MetaParam<T, N, typename std::enable_if<!UI::qt::ParamProxy<T>::IS_DEFAULT>::type>::_Param_proxy_constructor;

    MO_UI_QT_PARAMTERPROXY_METAParam(__COUNTER__)
}
//#include "detail/TParamProxyImpl.hpp"
