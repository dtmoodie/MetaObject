#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/ui/Qt/SignalProxy.hpp"
#include "THandler.hpp"
#include "UiUpdateHandler.hpp"

#include <qcombobox.h>
#include <qspinbox.h>

class QLineEdit;
class QCheckBox;
class QPushButton;
namespace mo
{
    struct EnumParam;
    struct WriteDirectory;
    struct ReadDirectory;
    struct WriteFile;
    struct ReadFile;
    namespace UI
    {
        namespace qt
        {
            // **********************************************************************************
            // *************************** Bool ************************************************
            // **********************************************************************************
            template <>
            class MO_EXPORTS THandler<bool, void> : public UiUpdateHandler
            {
                QCheckBox* chkBox;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const bool& data);
                void updateParam(bool& data);
                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent_);
                static bool uiUpdateRequired();
            };

            // **********************************************************************************
            // *************************** std::string ******************************************
            // **********************************************************************************

            template <>
            class MO_EXPORTS THandler<std::string, void> : public UiUpdateHandler
            {
                QLineEdit* lineEdit;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const std::string& data);
                void updateParam(std::string& data);

                std::vector<QWidget*> getUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** std::function<void(void)> **************************
            // **********************************************************************************

            template <>
            class MO_EXPORTS THandler<std::function<void(void)>, void> : public UiUpdateHandler
            {
                std::function<void(void)>* funcData;
                QPushButton* btn;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const std::function<void(void)>& data);
                void updateParam(std::function<void(void)>& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** floating point data **********************************
            // **********************************************************************************

            template <typename T>
            class THandler<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type>
                : public UiUpdateHandler
            {
                QDoubleSpinBox* box;

              public:
                THandler(IParamProxy& parent)
                    : box(nullptr)
                    , UiUpdateHandler(parent)
                {
                }

                void updateUi(const T& data)
                {
                    _updating = true;
                    if (box)
                        box->setValue(data);
                    _updating = false;
                }

                void updateParam(T& data)
                {
                    data = box->value();
                }

                std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    std::vector<QWidget*> output;
                    _parent_widget = parent;
                    if (box == nullptr)
                    {
                        box = new QDoubleSpinBox(parent);
                        box->setMaximumWidth(100);
                        box->setMinimum(std::numeric_limits<T>::min());
                        box->setMaximum(std::numeric_limits<T>::max());
                    }
                    box->connect(box, SIGNAL(valueChanged(double)), proxy, SLOT(on_update(double)));
                    output.push_back(box);
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** integers *********************************************
            // **********************************************************************************

            template <typename T>
            class THandler<T, typename std::enable_if<std::is_integral<T>::value, void>::type> : public UiUpdateHandler
            {
                QSpinBox* box;

              public:
                THandler(IParamProxy& parent)
                    : box(nullptr)
                    , UiUpdateHandler(parent)
                {
                }

                void updateUi(const T& data)
                {
                    if (box)
                        box->setValue(data);
                }
                void updateParam(T& data)
                {
                    if (box)
                        data = box->value();
                }
                std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    std::vector<QWidget*> output;
                    _parent_widget = parent;
                    if (box == nullptr)
                    {
                        box = new QSpinBox(parent);
                        box->setMaximumWidth(100);
                        box->setMaximum(static_cast<int>(
                            std::max<int64_t>(std::numeric_limits<T>::max(), std::numeric_limits<int>::max())));
                        box->setMinimum(static_cast<int>(
                            std::min<int64_t>(std::numeric_limits<T>::min(), std::numeric_limits<int>::min())));
                    }

                    box->connect(box, SIGNAL(valueChanged(int)), proxy, SLOT(on_update(int)));
                    box->connect(box, SIGNAL(editingFinished()), proxy, SLOT(on_update()));
                    output.push_back(box);
                    return output;
                }
            };
            // **********************************************************************************
            // *************************** Enums ************************************************
            // **********************************************************************************
            template <>
            class MO_EXPORTS THandler<EnumParam, void> : public UiUpdateHandler
            {
                QComboBox* enumCombo;
                bool _updating;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const EnumParam& data);
                void updateParam(EnumParam& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** Files ************************************************
            // **********************************************************************************

            template <>
            class MO_EXPORTS THandler<WriteDirectory, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const WriteDirectory& data);
                void updateParam(WriteDirectory& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent_);
            };
            template <>
            class MO_EXPORTS THandler<ReadDirectory, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const ReadDirectory& data);
                void updateParam(ReadDirectory& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent_);
            };
            template <>
            class MO_EXPORTS THandler<WriteFile, void> : public UiUpdateHandler
            {
                QPushButton* btn;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const WriteFile& data);
                void updateParam(WriteFile& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent_);
            };
            template <>
            class MO_EXPORTS THandler<ReadFile, void> : public UiUpdateHandler
            {
                QPushButton* btn;

              public:
                THandler(IParamProxy& parent);
                void updateUi(const ReadFile& data);
                void updateParam(ReadFile& data);
                std::vector<QWidget*> getUiWidgets(QWidget* parent_);
            };
        } // namespace qt
    }     // namespace UI
} // namespace mo
