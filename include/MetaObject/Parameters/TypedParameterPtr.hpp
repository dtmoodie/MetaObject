#pragma once
#include "ITypedParameter.hpp"
#include "MetaParameter.hpp"
namespace mo
{
    /*! The TypedParameterPtr class is a concrete implementation of ITypedParameter
     *  which implements wrapping of a raw pointer to user data.  This is used
     *  extensively inside of the OUTPUT macro as follows:
     *
     *  float user_data;
     *  TypedParameterPtr<float> user_param("float_data", &user_data);
     *  user_param.UpdateData(10);
     *
     *  This code snipit creates a user space variable 'user_data'
     *  which is wrapped for reflection purposes by 'user_param'
     */
    template<typename T>
    class TypedParameterPtr: virtual public ITypedParameter< T >
	{
	public:
        /*!
         * \brief TypedParameterPtr default constructor
         * \param name of the parameter
         * \param ptr_ to user owned data
         * \param type of parameter
         * \param ownsData_ cleanup on delete?
         */
        TypedParameterPtr(const std::string& name = "",
                          T* ptr_ = nullptr,
                          ParameterType type = Control_e,
                          bool ownsData_ = false);

        ~TypedParameterPtr();

        T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(), Context* ctx = nullptr, size_t* fn_ = nullptr);
        T*   GetDataPtr(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts_ = nullptr);
        T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(), Context* ctx = nullptr, size_t* fn = nullptr);
        T    GetData(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);
        bool GetData(T& value, boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(), Context* ctx = nullptr, size_t* fn = nullptr);
        bool GetData(T& value, size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

        /*virtual ITypedParameter<T>* UpdateData(const T& data,
                                               mo::time_t ts = -1 * mo::second,
                                               Context* ctx = nullptr,
                                               size_t fn = std::numeric_limits<size_t>::max(),
                                               ICoordinateSystem* cs = nullptr);*/

		virtual bool Update(IParameter* other);
		virtual std::shared_ptr<IParameter> DeepCopy() const;
        ITypedParameter<T>* UpdatePtr(T* ptr, bool ownsData_ = false);
	protected:
        virtual bool UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
		T* ptr;
		bool ownsData;
        static MetaParameter<T, 100> _meta_parameter;
	};
}
#include "detail/TypedParameterPtrImpl.hpp"
