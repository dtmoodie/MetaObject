#pragma once
#include "IHandler.hpp"
#include "IParamProxy.hpp"
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/core/TypeTable.hpp>

class QWidget;
class QObject;
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                Unspecialized handler
            // *****************************************************************************
            template <typename T, typename Enable = void>
            class THandler : public IHandler
            {
                IParamProxy& _parent;

              public:
                THandler(IParamProxy& parent)
                    : _parent(parent)
                {
                    MO_LOG(debug,
                           "Creating handler for default unspecialized param {}",
                           TypeTable::instance()->typeToName(TypeInfo(typeid(T))));
                }

                // Update user interface from parameter update
                void updateUi(const T& data)
                {
                }

                // update raw data from user interface
                void updateParam(T& data)
                {
                }

                // notify parent ParamProxy of the update, param proxy will lock parameter and call updateParam
                void onUiUpdate(QObject* sender)
                {
                    _parent.onUiUpdate(sender);
                }

                virtual std::vector<QWidget*> getUiWidgets(QWidget* parent)
                {
                    MO_LOG(debug,
                           "Creating widget for default unspecialized param {}",
                           TypeTable::instance()->typeToName(TypeInfo(typeid(T))));
                    return std::vector<QWidget*>();
                }
                inline void setUpdating(bool val = true)
                {
                }
                inline bool getUpdating() const
                {
                    return false;
                }
                static const bool UI_UPDATE_REQUIRED = false;
                static const bool IS_DEFAULT = true;
            };
        } // namespace qt
    }     // namespace UI
} // namespace mo
