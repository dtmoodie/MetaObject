#pragma once
#ifdef HAVE_QT5
#include "POD.hpp"
#include "IHandler.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            struct UiUpdateListener;
            template<class T, typename Enable> class THandler;
            // **********************************************************************************
            // *************************** std::pair ********************************************
            // **********************************************************************************

            template<typename T1> class THandler<std::pair<T1, T1>>  : public IHandler
            {
                std::pair<T1,T1>* pairData;
                THandler<T1> _handler1;
                THandler<T1> _handler2;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : 
                    pairData(nullptr), 
                    _currently_updating(false) 
                {
                }

                void UpdateUi( std::pair<T1, T1>* data)
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
                void onUiUpdate(QObject* sender)
                {
                    if(_currently_updating || !IHandler::getParamMtx())
                        return;
                    mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
                    _handler1.onUiUpdate(sender);
                    _handler2.onUiUpdate(sender);
                    if(_listener)
                        _listener->onUpdate(this);
                }
                virtual void setData(std::pair<T1, T1>* data_)
                {
                    pairData = data_;
                    if(data_)
                    {
                        _handler1.SetData(&data_->first);
                        _handler2.SetData(&data_->second);
                    }
                }
                std::pair<T1, T1>* getData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    auto out1 = _handler1.getUiWidgets(parent);
                    auto out2 = _handler2.getUiWidgets(parent);
                    out2.insert(out2.end(), out1.begin(), out1.end());
                    return out2;
                }
                virtual void setParamMtx(boost::recursive_mutex* mtx)
                {
                    IHandler::setParamMtx(mtx);
                    _handler1.setParamMtx(mtx);
                    _handler2.setParamMtx(mtx);
                }
                virtual void setUpdateListener(UiUpdateListener* listener)
                {
                    _handler1.setUpdateListener(listener);
                    _handler2.setUpdateListener(listener);
                }
            };

            template<typename T1, typename T2> class THandler<std::pair<T1, T2>>: public THandler<T1>, public THandler<T2>
            {
                std::pair<T1,T2>* pairData;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : pairData(nullptr) {}

                virtual void UpdateUi( std::pair<T1, T2>* data)
                {
                    
                }
                virtual void onUiUpdate(QObject* sender)
                {
                    if(IHandler::getParamMtx())
                    {
                        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
                        THandler<T1>::onUiUpdate(sender);
                        THandler<T2>::onUiUpdate(sender);
                    }
                }
                virtual void SetData(std::pair<T1, T2>* data_)
                {
                    pairData = data_;
                    THandler<T1>::SetData(&data_->first);
                    THandler<T2>::SetData(&data_->second);
                }
                std::pair<T1, T2>* getData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    
                    auto output = THandler<T1>::getUiWidgets(parent);
                    auto out2 = THandler<T2>::getUiWidgets(parent);
                    output.insert(output.end(), out2.begin(), out2.end());
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** std::vector ******************************************
            // **********************************************************************************
            template<typename T> class THandler<std::vector<T>, void> : public THandler < T, void >, public UiUpdateListener
            {
                std::vector<T>* vectorData;
                T _appendData;
                QSpinBox* index;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler(): 
                    index(new QSpinBox()), 
                    vectorData(nullptr), 
                    _currently_updating(false){
                    THandler<T>::setUpdateListener(this);
                }

                void updateUi( std::vector<T>* data){
                    if (data && data->size()){
                        mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
                        _currently_updating = true;
                        index->setMaximum(data->size());
                        if(index->value() < data->size())
                            THandler<T>::updateUi(&(*data)[index->value()]);
                        else
                            THandler<T>::updateUi(&_appendData);
                        _currently_updating = false;
                    }
                }

                void onUiUpdate(QObject* sender, int idx = 0){
                    if(_currently_updating || !IHandler::getParamMtx())
                        return;
                    if (sender == index && vectorData ){
                        if(vectorData->size() && idx < vectorData->size()){
                            mo::Mutex_t::scoped_lock lock(*IHandler::getParamMtx());
                            THandler<T>::setData(&(*vectorData)[idx]);
                            THandler<T>::onUiUpdate(sender);
                        }else{
                            THandler<T>::setData(&_appendData);
                            THandler<T>::onUiUpdate(sender);
                        }   
                    }
                }
                
                void setData(std::vector<T>* data_){
                    vectorData = data_;
                    if (vectorData){
                        if (data_->size()){
                            if (index && index->value() < vectorData->size())
                                THandler<T>::setData(&(*vectorData)[index->value()]);
                        }
                    }
                }
                
                std::vector<T>* getData(){
                    return vectorData;
                }
                
                std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    auto output = THandler<T>::getUiWidgets(parent);
                    index->setParent(parent);
                    index->setMinimum(0);
                    IHandler::proxy->connect(index, SIGNAL(valueChanged(int)), IHandler::proxy, SLOT(on_update(int)));
                    output.push_back(index);
                    return output;
                }
                
                void onUpdate(IHandler* handler){
                    if(THandler<T>::getData() == &_appendData && vectorData){
                        vectorData->push_back(_appendData);
                        THandler<T>::setData(&vectorData->back());
                        index->setMaximum(vectorData->size());
                    }
                }
            };
        }
    }
}
#endif