#pragma once
#ifdef HAVE_QT5
#include <MetaObject/Params/IParam.hpp>
#include "qwidget.h"
#include "qgridlayout.h"
#include "qpushbutton.h"
#include "qlabel.h"

namespace mo
{
    template<class T> class ITParam;
    template<class T> class ITRangedParam;
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

            template<typename T> 
            void SetMinMax(typename std::enable_if<has_minmax<THandler<T>>::value, THandler<T>>::type& handler, ITParam<T>* param)
            {
                /*auto rangedParam = dynamic_cast<ITRangedParam<T>*>(param);
                if (rangedParam)
                {
                    typename Handler<T>::min_max_type min, max;
                    rangedParam->GetRange(min, max);
                    handler.SetMinMax(min, max);
                }*/
            }

            template<typename T> 
            void SetMinMax(typename std::enable_if<!has_minmax<THandler<T>>::value, THandler<T>>::type& handler, ITParam<T>* param)
            {

            }
            
            template<typename T> 
            ParamProxy<T>::~ParamProxy()
            {
                //InvalidCallbacks::invalidate((void*)&paramHandler);
            }

            template<typename T> 
            void ParamProxy<T>::onUiUpdate()
            {
                //TODO Notify Param of update on the processing thread.
                Param->_modified = true;
                Param->OnUpdate(nullptr);
            }
            
            // Guaranteed to be called on the GUI thread thanks to the signal Connection configuration
            template<typename T> 
            void ParamProxy<T>::onParamUpdate(Context* ctx, IParam* param)
            {
                auto dataPtr = Param->Data();    
                if (dataPtr)
                {
                    if (THandler<T>::UiUpdateRequired())
                    {
                        paramHandler.UpdateUi(dataPtr);
                    }
                }
            }
            
            template<typename T> 
            void ParamProxy<T>::onParamDelete(IParam const* param)
            {
                if(param == Param)
                {
                    Param = nullptr;
                    paramHandler.SetParamMtx(nullptr);
                }
            }

            template<typename T> 
            ParamProxy<T>::ParamProxy(IParam* param)
            {
                SetParam(param);
            }
            
            template<typename T> 
            bool ParamProxy<T>::CheckParam(IParam* param)
            {
                return param == Param;
            }
            
            template<typename T> 
            QWidget* ParamProxy<T>::getParamWidget(QWidget* parent)
            {
                QWidget* output = new QWidget(parent);
                auto widgets = paramHandler.GetUiWidgets(output);
                SetMinMax<T>(paramHandler, Param);
                QGridLayout* layout = new QGridLayout(output);
                if (Param->getTypeInfo() == TypeInfo(typeid(std::function<void(void)>)))
                {
                    dynamic_cast<QPushButton*>(widgets[0])->setText(QString::fromStdString(Param->getName()));
                    layout->addWidget(widgets[0], 0, 0);
                }
                else
                {
                    QLabel* nameLbl = new QLabel(QString::fromStdString(Param->getName()), output);
                    nameLbl->setToolTip(QString::fromStdString(Param->getTypeInfo().name()));
                    layout->addWidget(nameLbl, 0, 0);
                    int count = 1;
                    output->setLayout(layout);
                    for (auto itr = widgets.rbegin(); itr != widgets.rend(); ++itr, ++count)
                    {
                        layout->addWidget(*itr, 0, count);
                    }
                    // Correct the tab order of the widgets
                    for(size_t i = widgets.size() - 1; i > 0; --i)
                    {
                        QWidget::setTabOrder(widgets[i], widgets[i - 1]);
                    }
                    paramHandler.UpdateUi(Param->GetDataPtr());
                }
                return output;
            }

            template<typename T> 
            bool ParamProxy<T>::SetParam(IParam* param)
            {
                if(param->getTypeInfo() != TypeInfo(typeid(T)))
                    return false;
                auto TParam = dynamic_cast<ITParam<T>*>(param);
                if (TParam)
                {
                    Param = TParam;
                    Param->mtx();
                    paramHandler.SetParamMtx(&Param->_mtx);
                    paramHandler.SetData(Param->GetDataPtr());
                    paramHandler.IHandler::GetOnUpdate() = std::bind(&ParamProxy<T>::onUiUpdate, this);
                    //Connection = Param->update_signal.connect(std::bind(&ParamProxy<T>::onParamUpdate, this, std::placeholders::_1), Signals::GUI, true, this);
                    //delete_Connection = Param->delete_signal.connect(std::bind(&ParamProxy<T>::onParamDelete, this));
                    return true;
                }
                return false;
            }
        }
    }
}
#endif // HAVE_QT5