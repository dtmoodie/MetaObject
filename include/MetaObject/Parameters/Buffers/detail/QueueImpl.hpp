#pragma once

namespace mo
{
namespace Buffer
{
template<class T>
Queue<T>::Queue(const std::string& name)
{

}

template<class T>
T* Queue<T>::GetDataPtr(mo::time_t ts, Context* ctx)
{

}

template<class T>
bool Queue<T>::GetData(T& value, mo::time_t ts, Context* ctx)
{

}

template<class T>
T Queue<T>::GetData(mo::time_t ts, Context* ctx)
{

}

template<class T>
ITypedParameter<T>* Queue<T>::UpdateData(T& data_, mo::time_t ts, Context* ctx)
{

}

template<class T>
ITypedParameter<T>* UpdateData(const T& data_, mo::time_t ts, Context* ctx)
{

}

template<class T>
ITypedParameter<T>* UpdateData(T* data_, mo::time_t ts, Context* ctx)
{

}

template<class T>
bool Update(IParameter* other, Context* ctx = nullptr)
{

}

template<class T>
std::shared_ptr<IParameter> DeepCopy() const
{

}
template<class T>
void SetSize(long long size)
{

}

template<class T>
long long GetSize()
{

}

template<class T>
void GetTimestampRange(mo::time_t& start, mo::time_t& end)
{

}


template<class T>
void onInputUpdate(Context* ctx, IParameter* param)
{

}
}
}
