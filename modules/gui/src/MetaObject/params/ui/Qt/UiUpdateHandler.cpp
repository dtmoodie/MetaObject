#include "UiUpdateHandler.hpp"

namespace mo
{
    namespace UI
    {
        namespace qt
        {

            UiUpdateHandler::UiUpdateHandler(IParamProxy& parent) : _parent(parent), _parent_widget(nullptr) {}
            void UiUpdateHandler::onUiUpdate(QObject* sender)
            {
                if (_updating)
                    return;
            }
            void UiUpdateHandler::onUiUpdate(QObject* sender, double val)
            {
                if (_updating)
                    return;
            }
            void UiUpdateHandler::onUiUpdate(QObject* sender, int val)
            {
                if (_updating)
                    return;
            }
            void UiUpdateHandler::onUiUpdate(QObject* sender, bool val)
            {
                if (_updating)
                    return;
            }
            void UiUpdateHandler::onUiUpdate(QObject* sender, QString val)
            {
                if (_updating)
                    return;
            }
            void UiUpdateHandler::onUiUpdate(QObject* sender, int row, int col)
            {
                if (_updating)
                    return;
            }
        }
    }
}
