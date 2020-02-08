#pragma once
#include "InputParam.hpp"
#include "OutputParam.hpp"

#include <MetaObject/thread/Mutex.hpp>

#include <mutex>

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
        ITInputParam(const ITInputParam&) = default;
        ITInputParam(ITInputParam&&) noexcept = default;
        ITInputParam& operator=(const ITInputParam&) = default;
        ITInputParam& operator=(ITInputParam&&) noexcept = default;
        ~ITInputParam() override = default;

        bool setInput(const std::shared_ptr<IParam>& input) override;
        bool setInput(IParam* input) override;

        bool acceptsInput(IParam* param) const override;
        bool acceptsType(const TypeInfo& type) const override;

        TypeInfo getTypeInfo() const override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void load(BinaryInputVisitor& ar) override;
        void save(BinaryOutputVisitor& ar) const override;
        void visit(StaticVisitor&) const override;

        IContainerPtr_t getData(const Header& desired = Header()) override;
        IContainerConstPtr_t getData(const Header& desired = Header()) const override;

        OptionalTime getTimestamp() const override;
        FrameNumber getFrameNumber() const override;

      protected:
        void updateDataImpl(const TContainerPtr_t& data, UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            TParam<T>::updateDataImpl(data, fg);
        }

        void onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags fgs) override
        {
            if (data->getType() == getTypeInfo())
            {
                auto typed = std::static_pointer_cast<TDataContainer<T>>(data);
                onInputUpdate(typed, param, fgs);
            }
        }

        virtual void onInputUpdate(TContainerPtr_t, IParam*, UpdateFlags);
    };

    template <class T>
    ITInputParam<T>::ITInputParam(const std::string& name)
        : TParam<T>(name, mo::ParamFlags::kINPUT)
        , IParam(name, mo::ParamFlags::kINPUT)
    {
    }

    template <class T>
    bool ITInputParam<T>::setInput(const std::shared_ptr<IParam>& input)
    {
        if (setInput(input.get()))
        {
            m_shared_input = input;
            return true;
        }
        return false;
    }

    template <class T>
    bool ITInputParam<T>::setInput(IParam* input)
    {
        Lock_t lock(this->mtx());
        if (input->getTypeInfo() == getTypeInfo())
        {
            if (InputParam::setInput(input))
            {
                // TParam<T>::updateData(input->getTypedData<T>());
                return true;
            }
        }
        else
        {
            MO_LOG(info,
                   "Unable to connect output '{}' of type <{}> to input '{}' of type <{}> due to mismatching types",
                   getTreeName(),
                   getTypeInfo().name(),
                   input->getTreeName(),
                   input->getTypeInfo().name());
        }

        return false;
    }

    template <class T>
    bool ITInputParam<T>::acceptsInput(IParam* param) const
    {
        if (param->checkFlags(mo::ParamFlags::kOUTPUT))
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
    TypeInfo ITInputParam<T>::getTypeInfo() const
    {
        return TParam<T>::getTypeInfo();
    }

    template <class T>
    OptionalTime ITInputParam<T>::getTimestamp() const
    {
        return InputParam::getTimestamp();
    }

    template <class T>
    FrameNumber ITInputParam<T>::getFrameNumber() const
    {
        return InputParam::getFrameNumber();
    }

    template <class T>
    void ITInputParam<T>::onInputUpdate(TContainerPtr_t data, IParam* param, UpdateFlags fg)
    {
        const auto header = data->getHeader();
        if (fg & mo::UpdateFlags::kBUFFER_UPDATED && param->checkFlags(mo::ParamFlags::kBUFFER))
        {
            TParam<T>::emitTypedUpdate(data, fg);
            emitUpdate(header, fg);
            return;
        }
        auto stream = param->getStream();
        if (stream == getStream())
        {
            updateDataImpl(data, UpdateFlags::kINPUT_UPDATED);
        }
    }
    /////////////////////////////////////////////////////////////////////////////////
    ///  ct::TArrayView specialization
    /////////////////////////////////////////////////////////////////////////////////

    template <class T>
    struct ITInputParam<ct::TArrayView<T>> : virtual public TParam<ct::TArrayView<T>>, virtual public InputParam
    {
      public:
        using Super_t = TParam<ct::TArrayView<T>>;
        using Vec_t = typename TDataContainer<std::vector<T>>::type;
        using TContainerPtr_t = typename TParam<ct::TArrayView<T>>::TContainerPtr_t;
        using ContainerConstPtr_t = typename TParam<ct::TArrayView<T>>::ContainerConstPtr_t;
        using TUpdateSlot_t = typename TParam<ct::TArrayView<T>>::TUpdateSlot_t;

        ITInputParam(const std::string& name)
            : TParam<ct::TArrayView<T>>(name, mo::ParamFlags::kINPUT)
            , IParam(name, mo::ParamFlags::kINPUT)
        {
        }

        ITInputParam(const ITInputParam&) = default;
        ITInputParam(ITInputParam&&) noexcept = default;
        ITInputParam& operator=(const ITInputParam&) = default;
        ITInputParam& operator=(ITInputParam&&) noexcept = default;

        ~ITInputParam() override = default;

        bool setInput(const std::shared_ptr<IParam>& input) override
        {
            if (setInput(input.get()))
            {
                m_shared_input = input;
                return true;
            }
            return false;
        }

        bool setInput(IParam* input) override
        {
            Lock_t lock(this->mtx());
            if (input->getTypeInfo() == getTypeInfo())
            {
                if (InputParam::setInput(input))
                {
                    Super_t::updateData(input->getTypedData<ct::TArrayView<T>>());
                    return true;
                }
            }
            else
            {
                if (input->getTypeInfo() == TypeInfo(typeid(Vec_t)))
                {
                    if (InputParam::setInput(input))
                    {
                        auto data = input->getData();
                        if (data)
                        {
                            Super_t::updateData(wrap(data));
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        bool acceptsInput(IParam* param) const override
        {
            if (param->checkFlags(mo::ParamFlags::kOUTPUT))
            {
                auto out_param = dynamic_cast<OutputParam*>(param);
                return out_param->providesOutput(getTypeInfo());
            }
            return param->getTypeInfo() == getTypeInfo();
        }

        // This is needed to ensure we have a unique final override for this type
        TypeInfo getTypeInfo() const
        {
            return Super_t::getTypeInfo();
        }

        bool acceptsType(const TypeInfo& type) const override
        {
            return type == getTypeInfo() || type == TypeInfo(typeid(Vec_t));
        }

        void load(ILoadVisitor& visitor) override
        {
            InputParam::load(visitor);
        }

        void save(ISaveVisitor& visitor) const override
        {
            InputParam::save(visitor);
        }

        void load(BinaryInputVisitor& ar) override
        {
            InputParam::load(ar);
        }

        void save(BinaryOutputVisitor& ar) const override
        {
            InputParam::save(ar);
        }

        void visit(StaticVisitor& ar) const override
        {
            InputParam::visit(ar);
        }

        IContainerPtr_t getData(const Header& desired = Header()) override
        {
            return InputParam::getData(desired);
        }

        IContainerConstPtr_t getData(const Header& desired = Header()) const override
        {
            return InputParam::getData(desired);
        }

        OptionalTime getTimestamp() const override
        {
            return InputParam::getTimestamp();
        }

        FrameNumber getFrameNumber() const override
        {
            return InputParam::getFrameNumber();
        }

        std::shared_ptr<TDataContainer<ct::TArrayView<T>>> getTypedData(const Header& desired = Header())
        {
            auto data = getData(desired);
            if (data)
            {
                return wrap(data);
            }
            return {};
        }

        std::shared_ptr<const TDataContainer<ct::TArrayView<T>>> getTypedData(const Header& desired = Header()) const
        {
            auto data = getData(desired);
            if (data)
            {
                return wrap(data);
            }
            return {};
        }

        bool getTypedData(ct::TArrayView<T>* data, const Header& desired = Header(), Header* retrieved = nullptr) const
        {
            auto container = getTypedData(desired);
            if (container)
            {
                if (retrieved)
                {
                    *retrieved = container->getHeader();
                }
                *data = container->data;
                return true;
            }
            return false;
        }

        bool getTypedData(ct::TArrayView<T>* data, const Header& desired = Header(), Header* retrieved = nullptr)
        {
            auto container = getTypedData(desired);
            if (container)
            {
                if (retrieved)
                {
                    *retrieved = container->getHeader();
                }
                *data = container->data;
                return true;
            }
            return false;
        }

      protected:
        typename TDataContainer<ct::TArrayView<T>>::Ptr_t wrap(typename IDataContainer::Ptr_t owning) const
        {
            typename TDataContainer<ct::TArrayView<T>>::Ptr_t wrapped;
            if (owning->getType() == getTypeInfo())
            {
                return std::static_pointer_cast<TDataContainer<ct::TArrayView<T>>>(owning);
            }
            if (owning->getType() == TypeInfo(typeid(Vec_t)))
            {
                auto typed = std::static_pointer_cast<TDataContainer<Vec_t>>(owning);
                wrapped = TParam<ct::TArrayView<T>>::create(ct::TArrayView<T>(typed->data.data(), typed->data.size()));
                wrapped->owning = owning;
            }

            return wrapped;
        }

        void updateDataImpl(const TContainerPtr_t& data, UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            TParam<ct::TArrayView<T>>::updateDataImpl(data, fg);
        }

        void onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags fgs) override
        {
            std::shared_ptr<TDataContainer<ct::TArrayView<T>>> wrapped;
            if (data->getType() == getTypeInfo())
            {
                wrapped = std::static_pointer_cast<TDataContainer<ct::TArrayView<T>>>(data);
            }
            else
            {
                if (data->getType() == TypeInfo(typeid(Vec_t)))
                {
                    auto typed = std::static_pointer_cast<TDataContainer<Vec_t>>(data);
                    wrapped = wrap(typed);
                }
            }
            if (wrapped)
            {
                onInputUpdate(wrapped, param, fgs);
            }
        }

        virtual void onInputUpdate(TContainerPtr_t data, IParam* param, UpdateFlags fg)
        {
            const auto header = data->getHeader();
            if (fg & mo::UpdateFlags::kBUFFER_UPDATED && param->checkFlags(mo::ParamFlags::kBUFFER))
            {
                TParam<ct::TArrayView<T>>::emitTypedUpdate(data, fg);
                emitUpdate(header, fg);
                return;
            }
            if (header.stream == getStream())
            {
                updateDataImpl(data, UpdateFlags::kINPUT_UPDATED);
            }
        }
    };
} // namespace mo
