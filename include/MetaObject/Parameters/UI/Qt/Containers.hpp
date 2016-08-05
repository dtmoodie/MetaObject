#pragma once

#include "POD.hpp"
namespace Parameters
{
    namespace UI
    {
        namespace qt
        {
            // **********************************************************************************
            // *************************** std::pair ********************************************
            // **********************************************************************************

            template<typename T1> class Handler<std::pair<T1, T1>>  : public IHandler
            {
                std::pair<T1,T1>* pairData;
                Handler<T1> _handler1;
                Handler<T1> _handler2;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                Handler() : pairData(nullptr), _currently_updating(false) {}

                virtual void UpdateUi( std::pair<T1, T1>* data)
                {
                    _currently_updating = true;
                    if(data)
                    {
                        _handler1.UpdateUi(&data->first);
                        _handler2.UpdateUi(&data->second);
                    }else
                    {
                        _handler1.UpdateUi(nullptr);
                        _handler2.UpdateUi(nullptr);
                    }
                    _currently_updating = false;
                }
                virtual void OnUiUpdate(QObject* sender)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    std::lock_guard<std::recursive_mutex> lock(*IHandler::GetParamMtx());
                    _handler1.OnUiUpdate(sender);
                    _handler2.OnUiUpdate(sender);
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(std::pair<T1, T1>* data_)
                {
                    pairData = data_;
                    if(data_)
                    {
                        _handler1.SetData(&data_->first);
                        _handler2.SetData(&data_->second);
                    }
                }
                std::pair<T1, T1>* GetData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    auto out1 = _handler1.GetUiWidgets(parent);
                    auto out2 = _handler2.GetUiWidgets(parent);
                    out2.insert(out2.end(), out1.begin(), out1.end());
                    return out2;
                }
                virtual void SetParamMtx(std::recursive_mutex* mtx)
                {
                    IHandler::SetParamMtx(mtx);
                    _handler1.SetParamMtx(mtx);
                    _handler2.SetParamMtx(mtx);
                }
                virtual void SetUpdateListener(UiUpdateListener* listener)
                {
                    _handler1.SetUpdateListener(listener);
                    _handler2.SetUpdateListener(listener);
                }
            };

            template<typename T1, typename T2> class Handler<std::pair<T1, T2>>: public Handler<T1>, public Handler<T2>
            {
                std::pair<T1,T2>* pairData;
            public:
                static const bool IS_DEFAULT = false;
                Handler() : pairData(nullptr) {}

                virtual void UpdateUi( std::pair<T1, T2>* data)
                {
                    
                }
                virtual void OnUiUpdate(QObject* sender)
                {
                    if(IHandler::GetParamMtx())
                    {
                        std::lock_guard<std::recursive_mutex> lock(*IHandler::GetParamMtx());
                        Handler<T1>::OnUiUpdate(sender);
                        Handler<T2>::OnUiUpdate(sender);
                    }
                }
                virtual void SetData(std::pair<T1, T2>* data_)
                {
                    pairData = data_;
                    Handler<T1>::SetData(&data_->first);
                    Handler<T2>::SetData(&data_->second);
                }
                std::pair<T1, T2>* GetData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    
                    auto output = Handler<T1>::GetUiWidgets(parent);
                    auto out2 = Handler<T2>::GetUiWidgets(parent);
                    output.insert(output.end(), out2.begin(), out2.end());
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** std::vector ******************************************
            // **********************************************************************************
            template<typename T> class Handler<std::vector<T>> : public Handler < T >, public UiUpdateListener
            {
                std::vector<T>* vectorData;
                T _appendData;
                QSpinBox* index;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                Handler(): index(new QSpinBox()), vectorData(nullptr), _currently_updating(false) 
                {
                    Handler<T>::SetUpdateListener(this);
                }
                virtual void UpdateUi( std::vector<T>* data)
                {
                    if (data && data->size())
                    {
                        std::lock_guard<std::recursive_mutex> lock(*IHandler::GetParamMtx());
                        _currently_updating = true;
                        index->setMaximum(data->size());
                        if(index->value() < data->size())
                            Handler<T>::UpdateUi(&(*data)[index->value()]);
                        else
                            Handler<T>::UpdateUi(&_appendData);
                        _currently_updating = false;
                    }
                }
                virtual void OnUiUpdate(QObject* sender, int idx = 0)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    if (sender == index && vectorData )
                    {
                        if(vectorData->size() && idx < vectorData->size())
                        {
                            std::lock_guard<std::recursive_mutex> lock(*IHandler::GetParamMtx());
                            Handler<T>::SetData(&(*vectorData)[idx]);
                            Handler<T>::OnUiUpdate(sender);
                        }else
                        {
                            Handler<T>::SetData(&_appendData);
                            Handler<T>::OnUiUpdate(sender);
                        }   
                    }
                }
                virtual void SetData(std::vector<T>* data_)
                {
                    vectorData = data_;
                    if (vectorData)
                    {
                        if (data_->size())
                        {
                            if (index && index->value() < vectorData->size())
                                Handler<T>::SetData(&(*vectorData)[index->value()]);
                        }
                    }
                }
                std::vector<T>* GetData()
                {
                    return vectorData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    auto output = Handler<T>::GetUiWidgets(parent);
                    index->setParent(parent);
                    index->setMinimum(0);
                    IHandler::proxy->connect(index, SIGNAL(valueChanged(int)), IHandler::proxy, SLOT(on_update(int)));
                    output.push_back(index);
                    return output;
                }
                virtual void OnUpdate(IHandler* handler)
                {
                    if(Handler<T>::GetData() == &_appendData && vectorData)
                    {
                        vectorData->push_back(_appendData);
                        Handler<T>::SetData(&vectorData->back());
                        index->setMaximum(vectorData->size());
                    }
                }
            };

        }
    }
}