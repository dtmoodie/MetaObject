#pragma once

#include <MetaObject/Parameters/UI/Qt.hpp>
#include "IParameterProxy.hpp"
#include "DefaultHandler.hpp"

namespace cv
{
    namespace cuda
    {
        class Stream;
    }
}

class QWidget;

namespace mo
{
    class IParameter;
    template<typename T> class ITypedParameter;
    template<typename T> class ITypedRangedParameter;
    namespace UI
    {
        namespace qt
        {
            template <typename T>
            class has_minmax
            {
                typedef char one;
                typedef long two;

                template <typename C> static one test(decltype(&C::SetMinMax));
                template <typename C> static two test(...);

            public:
                enum { value = sizeof(test<T>(0)) == sizeof(char) };
            };

            template<typename T> void SetMinMax(typename std::enable_if<has_minmax<Handler<T>>::value, Handler<T>>::type& handler, Parameters::ITypedParameter<T>* param)
            {
                auto rangedParam = dynamic_cast<ITypedRangedParameter<T>*>(param);
                if (rangedParam)
                {
                    typename Handler<T>::min_max_type min, max;
                    rangedParam->GetRange(min, max);
                    handler.SetMinMax(min, max);
                }
            }
            
            template<typename T> void SetMinMax(typename std::enable_if<!has_minmax<Handler<T>>::value, Handler<T>>::type& handler, Parameters::ITypedParameter<T>* param)
            {
                
            }
            
            // **********************************************************************************
            // *************************** ParameterProxy ***************************************
            // **********************************************************************************

           
            template<typename T> class ParameterProxy : public IParameterProxy
            {
                Handler<T> paramHandler;
                ITypedParameter<T>* parameter;
                std::shared_ptr<Signals::connection> connection;
                std::shared_ptr<Signals::connection> delete_connection;
            public:
                static const bool IS_DEFAULT = Handler<T>::IS_DEFAULT;
                ~ParameterProxy()
                {
                    InvalidCallbacks::invalidate((void*)&paramHandler);
                }
                void onUiUpdate()
                {
                    //TODO Notify parameter of update on the processing thread.
                    parameter->changed = true;
                    parameter->OnUpdate(nullptr);
                }
                // Guaranteed to be called on the GUI thread thanks to the signal connection configuration
                void onParamUpdate(cv::cuda::Stream* stream)
                {
                    auto dataPtr = parameter->Data();    
                    if (dataPtr)
                    {
                        if (Handler<T>::UiUpdateRequired())
                        {
                            paramHandler.UpdateUi(dataPtr);
                        }
                    }
                }
                void onParamDelete()
                {
                    parameter = nullptr;
                    connection.reset();
                    delete_connection.reset();
                    paramHandler.SetParamMtx(nullptr);
                }
                ParameterProxy(Parameters::Parameter* param)
                {
                    SetParameter(param);
                }
                virtual bool CheckParameter(Parameter* param)
                {
                    return param == parameter;
                }
                QWidget* GetParameterWidget(QWidget* parent)
                {
                    QWidget* output = new QWidget(parent);
                    auto widgets = paramHandler.GetUiWidgets(output);
                    SetMinMax<T>(paramHandler, parameter);
                    QGridLayout* layout = new QGridLayout(output);
                    if (parameter->GetTypeInfo() == Loki::TypeInfo(typeid(std::function<void(void)>)))
                    {
                        dynamic_cast<QPushButton*>(widgets[0])->setText(QString::fromStdString(parameter->GetName()));
                        layout->addWidget(widgets[0], 0, 0);
                    }
                    else
                    {
                        QLabel* nameLbl = new QLabel(QString::fromStdString(parameter->GetName()), output);
                        nameLbl->setToolTip(QString::fromStdString(parameter->GetTypeInfo().name()));
                        layout->addWidget(nameLbl, 0, 0);
                        int count = 1;
                        output->setLayout(layout);
                        for (auto itr = widgets.rbegin(); itr != widgets.rend(); ++itr, ++count)
                        {
                            layout->addWidget(*itr, 0, count);
                        }
                        // Correct the tab order of the widgets
                        for(int i = widgets.size() - 1; i > 0; --i)
                        {
                            QWidget::setTabOrder(widgets[i], widgets[i - 1]);
                        }
                        paramHandler.UpdateUi(parameter->Data());
                    }
                    return output;
                }
                virtual bool SetParameter(Parameter* param)
                {
                    auto typedParam = dynamic_cast<Parameters::ITypedParameter<T>*>(param);
                    if (typedParam)
                    {
                        parameter = typedParam;
                        paramHandler.SetParamMtx(&parameter->mtx());
                        paramHandler.SetData(parameter->Data());
                        paramHandler.IHandler::GetOnUpdate() = std::bind(&ParameterProxy<T>::onUiUpdate, this);
                        connection = parameter->update_signal.connect(std::bind(&ParameterProxy<T>::onParamUpdate, this, std::placeholders::_1), Signals::GUI, true, this);
                        delete_connection = parameter->delete_signal.connect(std::bind(&ParameterProxy<T>::onParamDelete, this));
                        return true;
                    }
                    return false;
                }
            };
            // **********************************************************************************
            // *************************** Constructor **********************************************
            // **********************************************************************************

            template<typename T> class Constructor
            {
            public:
                Factory()
                {
                    if(!ParameterProxy<T>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterCreator(TypeInfo(typeid(T)), std::bind(&Constructor<T>::Create, std::placeholders::_1));
                }
                static std::shared_ptr<IParameterProxy> Create(Parameters::Parameter* param)
                {
                    return std::shared_ptr<IParameterProxy>(new ParameterProxy<T>(param));
                }
            };
            template<typename T> class QtUiPolicy
            {
                static Constructor<T> _qt_ui_constructor;
            public:
                QtUiPolicy()
                {
                    (void)&_qt_ui_constructor;
                }
            };
            template<typename T> Constructor<T> QtUiPolicy<T>::_qt_ui_constructor;
        }
    }
}