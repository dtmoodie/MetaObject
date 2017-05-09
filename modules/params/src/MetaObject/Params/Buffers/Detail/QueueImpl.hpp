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
T* Queue<T>::GetDataPtr(mo::Time_t ts, Context* ctx)
{

}

template<class T>
bool Queue<T>::GetData(T& value, mo::Time_t ts, Context* ctx)
{

}

template<class T>
T Queue<T>::GetData(mo::Time_t ts, Context* ctx)
{

}

template<class T>
ITParam<T>* Queue<T>::UpdateData(T& data_, mo::Time_t ts, Context* ctx)
{

}

template<class T>
ITParam<T>* UpdateData(const T& data_, mo::Time_t ts, Context* ctx)
{

}

template<class T>
ITParam<T>* UpdateData(T* data_, mo::Time_t ts, Context* ctx)
{

}

template<class T>
bool Update(IParam* other, Context* ctx = nullptr)
{

}

template<class T>
std::shared_ptr<IParam> DeepCopy() const
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
void getTimestampRange(mo::Time_t& start, mo::Time_t& end)
{

}


template<class T>
void onInputUpdate(Context* ctx, IParam* param)
{

}
}
}
