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
        ~ITInputParam() override;

        bool setInput(const std::shared_ptr<IParam>& input);
        bool setInput(IParam* input);

        virtual bool acceptsInput(IParam* param) const;
        virtual bool acceptsType(const TypeInfo& type) const;

        virtual TypeInfo getTypeInfo() const override;

        virtual void visit(IReadVisitor&) override;
        virtual void visit(IWriteVisitor&) const override;
        virtual void visit(BinaryInputVisitor& ar) override;
        virtual void visit(BinaryOutputVisitor& ar) const override;

        virtual IContainerPtr_t getData(const Header& desired = Header()) override;
        virtual IContainerConstPtr_t getData(const Header& desired = Header()) const override;

      protected:
        void updateDataImpl(const TContainerPtr_t& data)
        {
            {
                mo::Lock_t lock(this->mtx());
                TParam<T>::_data = data;
            }
            emitUpdate(IDataContainer::Ptr(TParam<T>::_data), InputUpdated_e);
            TParam<T>::emitTypedUpdate(TParam<T>::_data, InputUpdated_e);
        }
        virtual void onInputUpdate(const IDataContainerPtr_t&, IParam*, UpdateFlags)
        {
        }
        virtual void onInputUpdate(TContainerPtr_t, IParam*, UpdateFlags);

      private:
        TUpdateSlot_t m_typed_update_slot;
        ConnectionPtr_t m_typed_connection;
    };

    template <class T>
    ITInputParam<T>::ITInputParam(const std::string& name)
        : TParam<T>(name)
    {
        m_typed_update_slot = std::bind(static_cast<void (ITInputParam<T>::*)(TContainerPtr_t, IParam*, UpdateFlags)>(
                                            &ITInputParam<T>::onInputUpdate),
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3);
    }

    template <class T>
    ITInputParam<T>::~ITInputParam()
    {
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
                TParam<T>::_data = param->getTypedData<T>();
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
        else
        {
            return param->getTypeInfo() == getTypeInfo();
        }
    }

    template <class T>
    bool ITInputParam<T>::acceptsType(const TypeInfo& type) const
    {
        return type == getTypeInfo();
    }

    template <class T>
    void ITInputParam<T>::visit(IReadVisitor& visitor)
    {
        InputParam::visit(visitor);
    }

    template <class T>
    void ITInputParam<T>::visit(IWriteVisitor& visitor) const
    {
        InputParam::visit(visitor);
    }

    template <class T>
    void ITInputParam<T>::visit(BinaryInputVisitor& ar)
    {
        InputParam::visit(ar);
    }

    template <class T>
    void ITInputParam<T>::visit(BinaryOutputVisitor& ar) const
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
