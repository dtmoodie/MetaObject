#pragma once
#include "MetaObject/core/detail/Forward.hpp"
namespace mo {
namespace Buffer {
    template <class T>
    Map<T>::Map(const std::string& name)
        : ITInputParam<T>(name) {
        this->appendFlags(Buffer_e);
    }

    template <class T>
    bool Map<T>::getData(InputStorage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_) {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());

        auto itr = search(ts);
        if (itr != _data_buffer.end()) {
            if (fn_) *fn_ = itr->first.fn;
            data          = (itr->second);
            this->_ts     = itr->first.ts;
            this->_fn     = itr->first.fn;
            this->_ctx    = itr->first.ctx;
            this->_cs     = itr->first.cs;
            return true;
        }
        return false;
    }

    template <class T>
    bool Map<T>::getData(InputStorage_t& value, size_t fn, Context* ctx, OptionalTime_t* ts) {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        auto                     itr = search(fn);
        if (itr != _data_buffer.end()) {
            if (ts) *ts = itr->first.ts;
            value       = (itr->second);
            this->_ts   = itr->first.ts;
            this->_fn   = itr->first.fn;
            this->_ctx  = itr->first.ctx;
            this->_cs   = itr->first.cs;
            return true;
        }
        return false;
    }

    template <class T>
    bool Map<T>::updateDataImpl(const Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs) {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        _data_buffer[{ ts, fn, cs, ctx }] = data;
        IParam::_modified = true;
        lock.unlock();
        IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        return true;
    }

    template <class T>
    void Map<T>::setFrameBufferCapacity(size_t size) {}

    template <class T>
    void Map<T>::setTimePaddingCapacity(mo::Time_t time) {}

    template <class T>
    boost::optional<size_t> Map<T>::getFrameBufferCapacity() {
        return {};
    }

    template <class T>
    OptionalTime_t Map<T>::getTimePaddingCapacity() {
        return {};
    }

    template <class T>
    size_t Map<T>::getSize() {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        return _data_buffer.size();
    }

    template <class T>
    bool Map<T>::getTimestampRange(mo::Time_t& start, mo::Time_t& end) {
        if (_data_buffer.size()) {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (_data_buffer.begin()->first.ts && _data_buffer.rbegin()->first.ts) {
                start = *_data_buffer.begin()->first.ts;
                end   = *_data_buffer.rbegin()->first.ts;
                return true;
            }
        }
        return false;
    }

    template <class T>
    bool Map<T>::getFrameNumberRange(size_t& start, size_t& end) {
        if (_data_buffer.size()) {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            start = _data_buffer.begin()->first.fn;
            end   = _data_buffer.rbegin()->first.fn;
            return true;
        }
        return false;
    }

    template <class T>
    typename std::map<SequenceKey, typename Map<T>::InputStorage_t>::iterator Map<T>::search(const OptionalTime_t& ts) {
        if (_data_buffer.size() == 0)
            return _data_buffer.end();
        if (!ts) {
            if (_data_buffer.size())
                return _data_buffer.rbegin().base();
            return _data_buffer.end();
        }
        return _data_buffer.find(*ts);
    }

    template <class T>
    typename std::map<SequenceKey, typename Map<T>::InputStorage_t>::iterator Map<T>::search(size_t fn) {
        if (_data_buffer.size() == 0)
            return _data_buffer.end();
        return _data_buffer.find(fn);
    }

    template <class T>
    void Map<T>::onInputUpdate(ConstStorageRef_t data, IParam* input, Context* ctx, OptionalTime_t ts, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs, UpdateFlags) {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        _data_buffer[{ ts, fn, cs, ctx }] = data;
        IParam::_modified = true;
        lock.unlock();
        IParam::_update_signal(this, ctx, ts, fn, cs, mo::BufferUpdated_e);
        ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
    }
}
}
