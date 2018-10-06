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
    struct ITInputParam : virtual public ITParam<T>, virtual public InputParam
    {
      public:
        using ContainerPtr_t = typename ITParam<T>::ContainerPtr_t;
        using ContainerConstPtr_t = typename ITParam<T>::ContainerConstPtr_t;
        using TUpdateSlot_t = typename ITParam<T>::TUpdateSlot_t;

        ITInputParam(const std::string& name = "");
        ~ITInputParam() override;

        bool setInput(const std::shared_ptr<IParam>& input);
        bool setInput(IParam* input);

        virtual bool acceptsInput(IParam* param) const;
        virtual bool acceptsType(const TypeInfo& type) const;

        IParam* getInputParam() const;
        OptionalTime_t getInputTimestamp();
        size_t getInputFrameNumber();

        virtual bool isInputSet() const;

      protected:
        virtual void onInputDelete(IParam const* param);
        virtual void onInputUpdate(ContainerPtr_t, IParam*, UpdateFlags);

      private:
        TUpdateSlot_t m_update_slot;
        TSlot<void(IParam const*)> m_delete_slot;
        std::shared_ptr<IParam> m_shared_input;
        ITParam<T>* m_input;
    };

    template <class T>
    ITInputParam<T>::ITInputParam(const std::string& name)
        : m_input(nullptr)
        , ITParam<T>(name)
    {
        m_update_slot = std::bind(&ITInputParam<T>::onInputUpdate,
                                  this,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3,
                                  std::placeholders::_4,
                                  std::placeholders::_5,
                                  std::placeholders::_6,
                                  std::placeholders::_7);
        m_delete_slot = std::bind(&ITInputParam<T>::onInputDelete, this, std::placeholders::_1);
    }

    template <class T>
    ITInputParam<T>::~ITInputParam()
    {
        if (m_input != nullptr)
        {
            m_input->unsubscribe();
        }
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
        Lock lock(this->mtx());
        if (param == nullptr)
        {
            if (m_input)
            {
                m_input->unsubscribe();
            }
            m_update_slot.clear();
            m_delete_slot.clear();
            m_input = nullptr;
            m_shared_input.reset();
            lock.unlock();
            IParam::emitUpdate(Header(), InputCleared_e);
            return true;
        }
        auto output_param = dynamic_cast<OutputParam*>(param);
        if ((output_param && output_param->providesOutput(getTypeInfo())) ||
            (param->getTypeInfo() == this->getTypeInfo()))
        {
            if (m_input)
            {
                m_input->unsubscribe();
            }
            if (output_param)
            {
                if (auto param_ = output_param->getOutputParam(TypeInfo(typeid(T))))
                {
                    this->_input = dynamic_cast<ITParam<T>*>(param_);
                }
            }
            else
            {
                this->_input = dynamic_cast<ITParam<T>*>(param);
            }
            if (this->_input)
            {
                param->subscribe();
                param->registerUpdateNotifier(&m_update_slot);
                param->registerDeleteNotifier(&m_delete_slot);
                lock.unlock();
                IParam::emitUpdate(Header(), InputSet_e);
                return true;
            }
        }
        return false;
    }

    template <class T>
    void ITInputParam<T>::onInputDelete(IParam const* param)
    {
        if (param != this->_input)
        {
            return;
        }
        else
        {
            Lock lock(this->mtx());
            this->_shared_input.reset();
            this->_input = nullptr;
            IParam::emitUpdate(Header(), InputCleared_e);
        }
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
    IParam* ITInputParam<T>::getInputParam() const
    {
        Lock lock(this->mtx());
        return m_input;
    }

    template <class T>
    OptionalTime_t ITInputParam<T>::getInputTimestamp()
    {
        Lock lock(this->mtx());
        if (m_input)
        {
            return m_input->getTimestamp();
        }
        else
        {
            THROW(debug) << "Input not set for " << getTreeName();
            return OptionalTime_t();
        }
    }

    template <class T>
    uint64_t ITInputParam<T>::getInputFrameNumber()
    {
        Lock lock(this->mtx());
        if (m_input)
        {
            return m_input->getFrameNumber();
        }
        else
        {
            THROW(debug) << "Input not set for " << getTreeName();
            return size_t(0);
        }
    }

    template <class T>
    bool ITInputParam<T>::isInputSet() const
    {
        return m_input != nullptr;
    }

    template <class T>
    void ITInputParam<T>::onInputUpdate(ContainerPtr_t data, IParam* param, UpdateFlags fg)
    {
        const auto header = data->getHeader();
        if (fg == mo::BufferUpdated_e && param->checkFlags(mo::ParamFlags::Buffer_e))
        {
            emitTypedUpdate(data, this, header, InputUpdated_e);
            emitUpdate(this, header, fg, InputUpdated_e);
            return;
        }
        if (header.ctx == this->m_ctx)
        {
            updateData(data);
        }
    }
}
