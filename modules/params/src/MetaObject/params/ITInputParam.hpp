#pragma once

#include "ITParam.hpp"
#include "InputParam.hpp"
#ifdef _MSC_VER
#pragma warning(disable : 4250)
#endif

namespace mo
{
    template <class T>
    class ITInputParam : virtual public ITParam<T>, virtual public InputParam
    {
      public:
        typedef typename ParamTraits<T>::Storage_t Storage_t;
        typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
        typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
        typedef typename ParamTraits<T>::Input_t Input_t;
        typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                   IParam*,
                                   Context*,
                                   OptionalTime_t,
                                   size_t,
                                   const std::shared_ptr<ICoordinateSystem>&,
                                   UpdateFlags);
        typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
        typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

        ITInputParam(const std::string& name = "", Context* ctx = nullptr);
        ~ITInputParam();
        bool setInput(std::shared_ptr<IParam> input);
        bool setInput(IParam* input);

        virtual bool acceptsInput(IParam* param) const;
        virtual bool acceptsType(const TypeInfo& type) const;

        IParam* getInputParam() const;
        OptionalTime_t getInputTimestamp();
        size_t getInputFrameNumber();

        virtual bool isInputSet() const;

        virtual bool getData(InputStorage_t& data,
                             const OptionalTime_t& ts = OptionalTime_t(),
                             Context* ctx = nullptr,
                             size_t* fn_ = nullptr);

        virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

      protected:
        bool updateDataImpl(const Storage_t& /*data*/,
                            const OptionalTime_t& /*ts*/,
                            Context* /*ctx*/,
                            size_t /*fn*/,
                            const std::shared_ptr<ICoordinateSystem>& /*cs*/) override
        {
            return true;
        }

        bool updateDataImpl(Storage_t&& /*data*/,
                            const OptionalTime_t& /*ts*/,
                            Context* /*ctx*/,
                            size_t /*fn*/,
                            const std::shared_ptr<ICoordinateSystem>& /*cs*/) override
        {
            return true;
        }

        virtual void onInputDelete(IParam const* param);
        virtual void onInputUpdate(ConstStorageRef_t,
                                   IParam*,
                                   Context*,
                                   OptionalTime_t,
                                   size_t,
                                   const std::shared_ptr<ICoordinateSystem>&,
                                   UpdateFlags);

        std::shared_ptr<IParam> _shared_input;
        ITParam<T>* _input;

      private:
        TUpdateSlot_t _update_slot;
        TSlot<void(IParam const*)> _delete_slot;
    };
}
#include "detail/ITInputParamImpl.hpp"
