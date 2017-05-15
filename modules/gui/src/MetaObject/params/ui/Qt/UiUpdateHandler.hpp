#pragma once
#include <MetaObject/detail/Export.hpp>
#include "IHandler.hpp"
#include "IParamProxy.hpp"
#include <QString>
namespace mo{
namespace UI{
namespace qt{
    template<typename T> class ParamProxy;
    class MO_EXPORTS UiUpdateHandler: public IHandler{
    public:
        UiUpdateHandler(IParamProxy& parent);
        virtual void onUiUpdate(QObject* sender);
        virtual void onUiUpdate(QObject* sender, double val);
        virtual void onUiUpdate(QObject* sender, int val);
        virtual void onUiUpdate(QObject* sender, bool val);
        virtual void onUiUpdate(QObject* sender, QString val);
        virtual void onUiUpdate(QObject* sender, int row, int col);
        static const bool UI_UPDATE_REQUIRED = true;
        static const bool IS_DEFAULT = false;
        inline void setUpdating(bool val = true){_updating = val;}
        inline bool getUpdating() const{return _updating;}
    protected:
        template<class T> friend class ParamProxy;
        IParamProxy& _parent;
        bool _updating;
        QWidget* _parent_widget;
    };
} // namespace mo::UI::qt
} // namespace mo::UI
} // namespace mo