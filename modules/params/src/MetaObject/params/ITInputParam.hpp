#pragma once
#include "ITParam.hpp"
#include "InputParam.hpp"
#include "OutputParam.hpp"

#ifdef _MSC_VER
#pragma warning(disable : 4250)
#endif

namespace mo
{
    template <class T>
    struct ITInputParam : virtual public TParam<T>, virtual public InputParam
    {
      public:
        using TContainerPtr_t = typename TParam<T>::TContainerPtr_t;
        using ContainerConstPtr_t = typename TParam<T>::ContainerConstPtr_t;
        using TUpdateSlot_t = typename TParam<T>::TUpdateSlot_t;

        ITInputParam(const std::string& name = "");
        ~ITInputParam() override = default;

        bool setInput(const std::shared_ptr<IParam>& input) override;
        bool setInput(IParam* input) override;

        virtual bool acceptsInput(IParam* param) const override;
        virtual bool acceptsType(const TypeInfo& type) const override;

        TypeInfo getTypeInfo() const override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void load(BinaryInputVisitor& ar) override;
        void save(BinaryOutputVisitor& ar) const override;
        void visit(StaticVisitor&) const override;

        IContainerPtr_t getData(const Header& desired = Header()) override;
        IContainerConstPtr_t getData(const Header& desired = Header()) const override;

      protected:
        void updateDataImpl(const TContainerPtr_t& data)
        {
            {
                mo::Lock_t lock(this->mtx());
                TParam<T>::updateDataImpl(data);
            }
            emitUpdate(IDataContainer::Ptr(TParam<T>::getData()), InputUpdated_e);
            TParam<T>::emitTypedUpdate(TParam<T>::getTypedData(), InputUpdated_e);
        }

        void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags) override
        {
        }

        virtual void onInputUpdate(TContainerPtr_t, IParam*, UpdateFlags);

      private:
        TUpdateSlot_t m_typed_update_slot;
        ConnectionPtr_t m_typed_connection;
    };

    template <class T>
    ITInputParam<T>::ITInputParam(const std::string& name)
        : TParam<T>(name, mo::ParamFlags::Input_e)
        , IParam(name, mo::ParamFlags::Input_e)
    {
        m_typed_update_slot = std::bind(static_cast<void (ITInputParam<T>::*)(TContainerPtr_t, IParam*, UpdateFlags)>(
                                            &ITInputParam<T>::onInputUpdate),
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3);
    }

    template <class T>
    bool ITInputParam<T>::setInput(const std::shared_ptr<IParam>& param)
    {
        if (setInput(param.get()))
        {
            m_shared_input = param;
            return true;
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::setInput(IParam* param)
    {
        Lock_t lock(this->mtx());
        if (param->getTypeInfo() == getTypeInfo())
        {
            if (InputParam::setInput(param))
            {
                m_typed_connection = param->registerUpdateNotifier(&m_typed_update_slot);
                TParam<T>::updateData(param->getTypedData<T>());
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::acceptsInput(IParam* param) const
    {
        if (param->checkFlags(mo::ParamFlags::Output_e))
        {
            auto out_param = dynamic_cast<OutputParam*>(param);
            return out_param->providesOutput(getTypeInfo());
        }
        return param->getTypeInfo() == getTypeInfo();
    }

    template <class T>
    bool ITInputParam<T>::acceptsType(const TypeInfo& type) const
    {
        return type == getTypeInfo();
    }

    template <class T>
    void ITInputParam<T>::load(ILoadVisitor& visitor)
    {
        InputParam::load(visitor);
    }

    template <class T>
    void ITInputParam<T>::save(ISaveVisitor& visitor) const
    {
        InputParam::save(visitor);
    }

    template <class T>
    void ITInputParam<T>::load(BinaryInputVisitor& ar)
    {
        InputParam::load(ar);
    }

    template <class T>
    void ITInputParam<T>::save(BinaryOutputVisitor& ar) const
    {
        InputParam::save(ar);
    }

    template <class T>
    void ITInputParam<T>::visit(StaticVisitor& ar) const
    {
        InputParam::visit(ar);
    }

    template <class T>
    ParamBase::IContainerPtr_t ITInputParam<T>::getData(const Header& desired)
    {
        return InputParam::getData(desired);
    }

    template <class T>
    ParamBase::IContainerConstPtr_t ITInputParam<T>::getData(const Header& desired) const
    {
        return InputParam::getData(desired);
    }

    template <class T>
    void ITInputParam<T>::onInputUpdate(TContainerPtr_t data, IParam* param, UpdateFlags fg)
    {
        const auto header = data->getHeader();
        if (fg == mo::BufferUpdated_e && param->checkFlags(mo::ParamFlags::Buffer_e))
        {
            TParam<T>::emitTypedUpdate(data, InputUpdated_e);
            emitUpdate(header, InputUpdated_e);
            return;
        }
        if (header.stream == getStream())
        {
            TParam<T>::updateData(data);
        }
    }

    template <class T>
    TypeInfo ITInputParam<T>::getTypeInfo() const
    {
        return TypeInfo(typeid(T));
    }
}
