#pragma once
#ifdef HAVE_QT5
#include "IHandler.hpp"
#include "POD.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            struct UiUpdateListener;
            template <class T, typename Enable>
            class THandler;
            // **********************************************************************************
            // *************************** std::pair ********************************************
            // **********************************************************************************

            template <typename T1, typename T2>
            class THandler<std::pair<T1, T2>> : public UiUpdateHandler
            {
                THandler<T1> _handler1;
                THandler<T2> _handler2;

              public:
                THandler(IParamProxy& parent) : UiUpdateHandler(parent), _handler1(parent), _handler2(parent) {}
                void updateUi(const std::pair<T1, T2>& data)
                {
                    _handler1.UpdateUi(data.first);
                    _handler2.UpdateUi(data.second);
                }
                void updateParam(std::pair<T1, T2>& data)
                {
                    _handler1(data.first);
                    _handler2(data.second);
                }
                std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    auto out1 = _handler1.getUiWidgets(parent);
                    auto out2 = _handler2.getUiWidgets(parent);
                    out2.insert(out2.end(), out1.begin(), out1.end());
                    return out2;
                }
            };

            // **********************************************************************************
            // *************************** std::vector ******************************************
            // **********************************************************************************
            template <typename T>
            class THandler<std::vector<T>, void> : public UiUpdateHandler
            {
                QSpinBox* index;
                THandler<T, void> _data_handler;

              public:
                THandler(IParamProxy& parent) : index(new QSpinBox()), UiUpdateHandler(parent), _data_handler(parent)
                {
                    index->setMinimum(0);
                }

                void updateUi(const std::vector<T>& data)
                {
                    if (data.size())
                    {
                        index->setMaximum(static_cast<int>(data.size()));
                        if (index->value() < data.size())
                            _data_handler.updateUi(data[index->value()]);
                    }
                }

                void updateParam(std::vector<T>& data)
                {
                    auto idx = index->value();
                    if (idx < data.size())
                    {
                        _data_handler.updateParam(data[idx]);
                    }
                    else if (idx == data.size())
                    {
                        T append_data;
                        _data_handler.updateParam(append_data);
                        data.push_back(append_data);
                    }
                }

                std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    auto output = _data_handler.getUiWidgets(parent);
                    index->setParent(parent);
                    index->setMinimum(0);
                    IHandler::proxy->connect(index, SIGNAL(valueChanged(int)), IHandler::proxy, SLOT(on_update(int)));
                    output.push_back(index);
                    return output;
                }
            };
        }
    }
}
#endif
