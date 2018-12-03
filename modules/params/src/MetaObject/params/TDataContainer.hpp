#pragma once
#include "Header.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/params/IDynamicVisitor.hpp>

#include <ct/reflect/cerealize.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/boost/optional.hpp>

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

        virtual void visit(IReadVisitor&) override;
        virtual void visit(IWriteVisitor&) const override;
        virtual void visit(BinaryInputVisitor& ar) override;
        virtual void visit(BinaryOutputVisitor& ar) const override;

        virtual const Header& getHeader() const;

        operator std::shared_ptr<T>();
        operator std::shared_ptr<const T>() const;

        operator T*();
        operator const T*() const;

        T data;
        Header header;
    };

    ////////////////////////////////////////////////////////////////////////
    ///  Implementation
    ////////////////////////////////////////////////////////////////////////
    template <class T>
    TDataContainer<T>::TDataContainer(const T& data_)
        : data(data_)
    {
    }

    template <class T>
    TDataContainer<T>::TDataContainer(T&& data_)
        : data(std::move(data_))
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
    void TDataContainer<T>::visit(IReadVisitor& visitor)
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainer<T>::visit(IWriteVisitor& visitor) const
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainer<T>::visit(BinaryInputVisitor& ar)
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
    }

    template <class T>
    void TDataContainer<T>::visit(BinaryOutputVisitor& ar) const
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
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

    template <class T>
    TDataContainer<T>::operator T*()
    {
        return &data;
    }

    template <class T>
    TDataContainer<T>::operator const T*() const
    {
        return &data;
    }
}
