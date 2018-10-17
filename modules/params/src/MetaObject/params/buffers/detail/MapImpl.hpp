#pragma once
#include "MetaObject/core/detail/Forward.hpp"
namespace mo
{
    namespace Buffer
    {
        template <class T>
        Map<T>::Map(const std::string& name)
            : ITInputParam<T>(name)
        {
            this->appendFlags(ParamFlags::Buffer_e);
        }

        template <class T>
        bool Map<T>::getData(InputStorage_t& data, const OptionalTime& ts, Context* ctx, size_t* fn_)
        {
            Lock lock(IParam::mtx());

            auto itr = search(ts);
            if (itr != _data_buffer.end())
            {
                if (fn_)
                {
                    *fn_ = itr->first.fn;
                }
                else
                {
                }
                data = (itr->second);
                this->_ts = itr->first.ts;
                this->_fn = itr->first.fn;
                this->_ctx = itr->first.ctx;
                this->_cs = itr->first.cs;
                return true;
            }
            else
            {
            }
            return false;
        }

        template <class T>
        bool Map<T>::getData(InputStorage_t& value, size_t fn, Context* ctx, OptionalTime* ts)
        {
            Lock lock(IParam::mtx());
            auto itr = search(fn);
            if (itr != _data_buffer.end())
            {
                if (ts)
                {
                    *ts = itr->first.ts;
                }
                else
                {
                }
                value = itr->second;
                this->_ts = itr->first.ts;
                this->_fn = itr->first.fn;
                this->_ctx = itr->first.ctx;
                this->_cs = itr->first.cs;
                return true;
            }
            else
            {
                return false;
            }
        }

        template <class T>
        bool Map<T>::updateDataImpl(const Storage_t& data,
                                    const OptionalTime& ts,
                                    Context* ctx,
                                    size_t fn,
                                    const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock lock(IParam::mtx());
            _data_buffer[{ts, fn, cs, ctx}] = data;
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            TParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        bool Map<T>::updateDataImpl(Storage_t&& data,
                                    const OptionalTime& ts,
                                    Context* ctx,
                                    size_t fn,
                                    const std::shared_ptr<ICoordinateSystem>& cs)
        {
            Lock lock(IParam::mtx());
            auto itr = _data_buffer.emplace(SequenceKey(ts, fn, cs, ctx), std::move(data));
            this->modified(true);
            lock.unlock();
            IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            TParamImpl<T>::emitTypedUpdate(itr.first->second, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            return true;
        }

        template <class T>
        void Map<T>::setFrameBufferCapacity(size_t size)
        {
        }

        template <class T>
        void Map<T>::setTimePaddingCapacity(mo::Time time)
        {
        }

        template <class T>
        boost::optional<size_t> Map<T>::getFrameBufferCapacity()
        {
            return {};
        }

        template <class T>
        OptionalTime Map<T>::getTimePaddingCapacity()
        {
            return {};
        }

        template <class T>
        size_t Map<T>::getSize()
        {
            Lock lock(IParam::mtx());
            return _data_buffer.size();
        }

        template <class T>
        bool Map<T>::getTimestampRange(mo::Time& start, mo::Time& end)
        {
            if (_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                if (_data_buffer.begin()->first.ts && _data_buffer.rbegin()->first.ts)
                {
                    start = *_data_buffer.begin()->first.ts;
                    end = *_data_buffer.rbegin()->first.ts;
                    return true;
                }
                else
                {
                }
            }
            else
            {
            }
            return false;
        }

        template <class T>
        bool Map<T>::getFrameNumberRange(size_t& start, size_t& end)
        {
            if (_data_buffer.size())
            {
                Lock lock(IParam::mtx());
                start = _data_buffer.begin()->first.fn;
                end = _data_buffer.rbegin()->first.fn;
                return true;
            }
            else
            {
                return false;
            }
        }

        template <class T>
        typename std::map<SequenceKey, typename Map<T>::InputStorage_t>::iterator Map<T>::search(const OptionalTime& ts)
        {
            if (_data_buffer.size() == 0)
            {
                return _data_buffer.end();
            }
            else
            {
                if (!ts)
                {
                    if (_data_buffer.size())
                    {
                        return (--this->_data_buffer.end());
                    }
                    else
                    {
                        return _data_buffer.end();
                    }
                }
                else
                {
                }
            }

            return _data_buffer.find(*ts);
        }

        template <class T>
        typename std::map<SequenceKey, typename Map<T>::InputStorage_t>::iterator Map<T>::search(size_t fn)
        {
            if (_data_buffer.size() == 0)
            {
                return _data_buffer.end();
            }
            else
            {
                return _data_buffer.find(fn);
            }
        }

        template <class T>
        void Map<T>::onInputUpdate(ConstStorageRef_t data,
                                   IParam* input,
                                   Context* ctx,
                                   OptionalTime ts,
                                   size_t fn,
                                   const std::shared_ptr<ICoordinateSystem>& cs,
                                   UpdateFlags)
        {
            Lock lock(IParam::mtx());
            _data_buffer[{ts, fn, cs, ctx}] = data;
            this->modified(true);
            lock.unlock();
            IParam::emitUpdate(ts, ctx, fn, cs, mo::BufferUpdated_e);
            TParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        }
    }
}
