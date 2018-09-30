#pragma once
#include "Header.hpp"
#include "IDataContainer.hpp"
namespace mo
{
    template <class T>
    struct TDataContainer : public IDataContainer
    {
        using Ptr = std::shared_ptr<TDataContainer<T>>;
        using ConstPtr = std::shared_ptr<const TDataContainer<T>>;

        TDataContainer(const T& data = T());
        TDataContainer(T&& data);

        virtual ~TDataContainer() override;

        virtual TypeInfo getType() const;

        virtual void visit(IReadVisitor*);
        virtual void visit(IWriteVisitor*) const;

        virtual const Header& getHeader() const;

        operator std::shared_ptr<T>();
        operator std::shared_ptr<const T>() const;

        T data;
        Header header;
    };

    ////////////////////////////////////////////////////////////////////////
    ///  Implementation
    ////////////////////////////////////////////////////////////////////////
    template <class T>
    TDataContainer<T>::TDataContainer(const T& data_) : data(data_)
    {
    }

    template <class T>
    TDataContainer<T>::TDataContainer(T&& data_) : data(std::move(data_))
    {
    }

    template <class T>
    TDataContainer<T>::~TDataContainer()
    {
    }

    template <class T>
    TypeInfo TDataContainer<T>::getType() const
    {
        return TypeInfo::create<T>();
    }

    template <class T>
    void TDataContainer<T>::visit(IReadVisitor* visitor)
    {
        (*visitor)(&data);
    }

    template <class T>
    void TDataContainer<T>::visit(IWriteVisitor* visitor) const
    {
        (*visitor)(&data);
    }

    template <class T>
    const Header& TDataContainer<T>::getHeader() const
    {
        return header;
    }

    template <class T>
    TDataContainer<T>::operator std::shared_ptr<T>()
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    TDataContainer<T>::operator std::shared_ptr<const T>() const
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<T>(&data, [owning_ptr](T*) {});
    }
}
