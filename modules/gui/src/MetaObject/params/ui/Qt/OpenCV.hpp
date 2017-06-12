#pragma once
#if defined( HAVE_QT5 ) && defined(HAVE_OPENCV)
#include "POD.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>

#include <qtablewidget.h>
#include "qheaderview.h"

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            template<typename T, typename Enable> class THandler;

            // **********************************************************************************
            // *************************** cv::Rect *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Rect_<T>, void> : public UiUpdateHandler{
                THandler<T> _x_handler;
                THandler<T> _y_handler;
                THandler<T> _width_handler;
                THandler<T> _height_handler;
            public:
                static const bool IS_DEFAULT = false;
                THandler(IParamProxy& parent): UiUpdateHandler(parent), _x_handler(parent), _y_handler(parent),
                    _width_handler(parent), _height_handler(parent){}

                virtual void updateUi( const ::cv::Rect_<T>& data){
                    _x_handler.updateUi(data.x);
                    _y_handler.updateUi(data.y);
                    _width_handler.updateUi(data.width);
                    _height_handler.updateUi(data.height);
                }

                void onUiUpdate(QObject* sender) { _parent.onUiUpdate(sender); }

                void updateParam(::cv::Rect_<T>& data) {
                    _x_handler.updateParam(data.x);
                    _y_handler.updateParam(data.y);
                    _width_handler.updateParam(data.width);
                    _height_handler.updateParam(data.height);
                }

                std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    std::vector<QWidget*> output;
                    auto out1 = _x_handler.getUiWidgets(parent);
                    auto out2 = _y_handler.getUiWidgets(parent);
                    auto out3 = _width_handler.getUiWidgets(parent);
                    auto out4 = _height_handler.getUiWidgets(parent);
                    output.insert(output.end(), out1.begin(), out1.end());
                    output.insert(output.end(), out2.begin(), out2.end());
                    output.insert(output.end(), out3.begin(), out3.end());
                    output.insert(output.end(), out4.begin(), out4.end());
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** cv::Range *********************************************
            // **********************************************************************************

            template<> class THandler<cv::Range, void> : public UiUpdateHandler{
                THandler<int> _start_handler;
                THandler<int> _end_handler;
            public:
                THandler(IParamProxy& parent) :
                    _start_handler(parent),
                    _end_handler(parent),
                    UiUpdateHandler(parent){}

                void updateUi( const cv::Range& data){
                    _start_handler.updateUi(data.start);
                    _end_handler.updateUi(data.end);
                }
                void updateParam(cv::Range& data){
                    _start_handler.updateParam(data.start);
                    _end_handler.updateParam(data.end);
                }

                std::vector < QWidget*> getUiWidgets(QWidget* parent_){
                    std::vector<QWidget*> output;
                    auto out1 = _start_handler.getUiWidgets(parent_);
                    auto out2 = _end_handler.getUiWidgets(parent_);
                    output.insert(output.end(), out1.begin(), out1.end());
                    output.insert(output.end(), out2.begin(), out2.end());
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** cv::Matx *********************************************
            // **********************************************************************************
            template<typename T, int ROW, int COL> class THandler<typename ::cv::Matx<T, ROW, COL>, void> : public UiUpdateHandler{
                QTableWidget* table;
                std::vector<QTableWidgetItem*> items;
                ::cv::Matx<T, ROW, COL>* matData;
            public:
                THandler(IParamProxy& parent) : table(nullptr), matData(nullptr), UiUpdateHandler(parent){
                    table = new QTableWidget();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    items.reserve(ROW*COL);
                    if(COL == 1){
                        table->setColumnCount(ROW);
                        table->setRowCount(1);
                    }else{
                        table->setColumnCount(COL);
                        table->setRowCount(ROW);
                    }
                    for (int i = 0; i < ROW; ++i){
                        for (int j = 0; j < COL; ++j){
                            if(COL != 1) table->setColumnWidth(j, 40);
                            else table->setColumnWidth(i, 40);
                            QTableWidgetItem* item = new QTableWidgetItem();
                            items.push_back(item);
                            if(COL == 1) table->setItem(0, i, item);
                            else table->setItem(i, j, item);
                        }
                    }
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }

                void updateUi(const ::cv::Matx<T, ROW, COL>& data){
                    for (int i = 0; i < ROW; ++i){
                        for (int j = 0; j < COL; ++j){
                            items[i*COL + j]->setToolTip(QString::number((data)(i,j)));
                            items[i*COL + j]->setData(Qt::EditRole, (data)(i, j));
                        }
                    }
                }
                void updateParam(::cv::Matx<T, ROW, COL>& data){
                    for(int row = 0; row < ROW; ++row){
                        for(int col = 0; col < COL; ++col){
                            if (typeid(T) == typeid(float))
                                data(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toFloat();
                            if (typeid(T) == typeid(double))
                                data(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toDouble();
                            if (typeid(T) == typeid(int))
                                data(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toInt();
                            if (typeid(T) == typeid(unsigned int))
                                data(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toUInt();
                        }
                    }
                }
                std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
            template<typename T, int ROW> class THandler<typename ::cv::Vec<T, ROW>, void> : public THandler<::cv::Matx<T,ROW,1>>{
            public:
                THandler(IParamProxy& parent):
                    THandler<::cv::Matx<T,ROW,1>>(parent){}
            };
            template<typename T> class THandler<typename ::cv::Scalar_<T>, void> : public THandler<::cv::Vec<T, 4>>{
            public:
                THandler(IParamProxy& parent):
                    THandler<cv::Vec<T,4>>(parent){}
            };

            // **********************************************************************************
            // *************************** cv::Point_ *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Point_<T>, void> : public UiUpdateHandler
            {
                QTableWidget* table;
                QTableWidgetItem* first;
                QTableWidgetItem* second;
                ::cv::Point_<T>* ptData;
            public:
                THandler(IParamProxy& parent):UiUpdateHandler(parent), ptData(nullptr){
                    table = new QTableWidget();
                    first = new QTableWidgetItem();
                    second = new QTableWidgetItem();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    table->setItem(0, 0, first);
                    table->setItem(0, 1, second);
                    table->setRowCount(1);
                    table->setColumnCount(2);
                    table->setColumnWidth(0, 40);
                    table->setColumnWidth(1, 40);
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }
                void updateUi(const ::cv::Point_<T>& data){
                    if (table){
                        first = new QTableWidgetItem();
                        second = new QTableWidgetItem();
                        first->setData(Qt::EditRole, data.x);
                        second->setData(Qt::EditRole, data.y);
                        first->setToolTip(QString::number(data.x));
                        second->setToolTip(QString::number(data.y));
                        table->setItem(0, 0, first);
                        table->setItem(0, 1, second);
                        table->update();
                    }
                }
                void updateParam(::cv::Point_<T>& data){
                    if (typeid(T) == typeid(double)){
                        data.x = (T)first->data(Qt::EditRole).toDouble();
                        data.y = (T)second->data(Qt::EditRole).toDouble();
                    }
                    if (typeid(T) == typeid(float)){
                        data.x = (T)first->data(Qt::EditRole).toFloat();
                        data.y = (T)second->data(Qt::EditRole).toFloat();
                    }
                    if (typeid(T) == typeid(int)){
                        data.x = (T)first->data(Qt::EditRole).toInt();
                        data.y = (T)second->data(Qt::EditRole).toInt();
                    }
                    if (typeid(T) == typeid(unsigned int)){
                        data.x = (T)first->data(Qt::EditRole).toUInt();
                        data.y = (T)second->data(Qt::EditRole).toUInt();
                    }
                }
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
            // **********************************************************************************
            // *************************** cv::Point3_ *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Point3_<T>, void> : public UiUpdateHandler{
                QTableWidget* table;
                QTableWidgetItem* first;
                QTableWidgetItem* second;
                QTableWidgetItem* third;
            public:
                THandler(IParamProxy& parent):UiUpdateHandler(parent){
                    table = new QTableWidget();
                    first = new QTableWidgetItem();
                    second = new QTableWidgetItem();
                    third = new QTableWidgetItem();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    table->setItem(0, 0, first);
                    table->setItem(0, 1, second);
                    table->setItem(0, 2, third);
                    table->setRowCount(1);
                    table->setColumnCount(3);
                    table->setColumnWidth(0, 40);
                    table->setColumnWidth(1, 40);
                    table->setColumnWidth(2, 40);
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }
                void updateUi(const ::cv::Point3_<T>& data){
                        first = new QTableWidgetItem();
                        second = new QTableWidgetItem();
                        third = new QTableWidgetItem();
                        first->setData(Qt::EditRole, data.x);
                        second->setData(Qt::EditRole, data.y);
                        third->setData(Qt::EditRole, data.z);
                        first->setToolTip(QString::number(data.x));
                        second->setToolTip(QString::number(data.y));
                        third->setToolTip(QString::number(data.z));
                        table->setItem(0, 0, first);
                        table->setItem(0, 1, second);
                        table->setItem(0, 2, third);
                }
                void updateParam(::cv::Point3_<T>& data){
                    if (typeid(T) == typeid(double)){
                        data.x = (T)first->data(Qt::EditRole).toDouble();
                        data.y = (T)second->data(Qt::EditRole).toDouble();
                        data.z = (T)third->data(Qt::EditRole).toDouble();
                    }
                    if (typeid(T) == typeid(float)){
                        data.x = (T)first->data(Qt::EditRole).toFloat();
                        data.y = (T)second->data(Qt::EditRole).toFloat();
                        data.z = (T)third->data(Qt::EditRole).toFloat();
                    }
                    if (typeid(T) == typeid(int)){
                        data.x = (T)first->data(Qt::EditRole).toInt();
                        data.y = (T)second->data(Qt::EditRole).toInt();
                        data.z = (T)third->data(Qt::EditRole).toInt();
                    }
                    if (typeid(T) == typeid(unsigned int)){
                        data.x = (T)first->data(Qt::EditRole).toUInt();
                        data.y = (T)second->data(Qt::EditRole).toUInt();
                        data.z = (T)third->data(Qt::EditRole).toUInt();
                    }
                }

                std::vector<QWidget*> getUiWidgets(QWidget* parent){
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
        }
    }
}
#endif
