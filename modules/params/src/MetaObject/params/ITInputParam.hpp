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
        using ContainerPtr_t = typename ITParam<T>::ContainerPtr_t;
        using TUpdateSlot_t = typename ITParam<T>::TUpdateSlot_t;

        ITInputParam(const std::string& name = "")
            : ITParam<T>(name, ParamFlags::Input_e)
        {
            _update_slot = std::bind(&ITInputParam<T>::onInputUpdate,
                                     this,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4,
                                     std::placeholders::_5,
                                     std::placeholders::_6,
                                     std::placeholders::_7);
            _delete_slot = std::bind(&ITInputParam<T>::onInputDelete, this, std::placeholders::_1);
        }

        ~ITInputParam()
        {
            if (_input)
            {
                _input->unsubscribe();
            }
        }

        bool setInput(const std::shared_ptr<IParam>& input) override
        {
            if (setInput(input.get()))
            {
                Lock lock(mtx());
                _shared_input = input;
                return true;
            }
            return false;
        }

        bool setInput(IParam* input) override
        {
            Lock lock(this->mtx());
            if (input == nullptr)
            {
                if (_input)
                {
                    _input->unsubscribe();
                }
                _update_slot.clear();
                _delete_slot.clear();
                _input = nullptr;
                _shared_input.reset();
                lock.unlock();
                IParam::emitUpdate({}, InputCleared_e);
                return true;
            }
            auto output_param = dynamic_cast<OutputParam*>(input);
            if ((output_param && output_param->providesOutput(getTypeInfo())) ||
                (input->getTypeInfo() == this->getTypeInfo()))
            {
                if (_input)
                {
                    _input->unsubscribe();
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
                    this->_input = dynamic_cast<ITParam<T>*>(input);
                }
                if (this->_input)
                {
                    this->_input->subscribe();
                    this->_input->registerUpdateNotifier(&_update_slot);
                    this->_input->registerDeleteNotifier(&_delete_slot);
                    lock.unlock();
                    auto header = this->_input->getHeader();
                    IParam::emitUpdate(header, InputSet_e);
                    return true;
                }
            }
            return false;
        }

        bool acceptsInput(IParam* param) const override
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

        bool acceptsType(const TypeInfo& type) const override
        {
            return type == getTypeInfo();
        }

        IParam* getInputParam() const
        {
            Lock lock(this->mtx());
            return _input;
        }

        OptionalTime_t getInputTimestamp()
        {
            Lock lock(this->mtx());
            if (_input)
            {
                return _input->getTimestamp();
            }
            else
            {
                THROW(debug) << "Input not set for " << getTreeName();
                return OptionalTime_t();
            }
        }

        uint64_t getInputFrameNumber() override
        {
            Lock lock(this->mtx());
            if (_input)
            {
                return _input->getFrameNumber();
            }
            else
            {
                THROW(debug) << "Input not set for " << getTreeName();
                return size_t(0);
            }
        }

        bool isInputSet() const override
        {
            return _input != nullptr;
        }

      protected:
        virtual void onInputDelete(IParam const* param)
        {
            if (param == this->_input)
            {
                Lock lock(this->mtx());
                this->_shared_input.reset();
                this->_input = nullptr;
                IParam::emitUpdate({}, InputCleared_e);
            }
        }

        virtual void onInputUpdate(ContainerPtr_t data, const IParam*, UpdateFlags)
        {
            IParam::emitUpdate(data->header, InputUpdated_e);
        }

        std::shared_ptr<IParam> _shared_input;
        ITParam<T>* _input = nullptr;

      private:
        TUpdateSlot_t _update_slot;
        TSlot<void(IParam const*)> _delete_slot;
    };
}
