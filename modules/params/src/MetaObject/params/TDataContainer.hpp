#pragma once
#include "Header.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/time.hpp>
#include <ct/reflect/cerealize.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <boost/optional.hpp>

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

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void load(BinaryInputVisitor& ar) override;
        void save(BinaryOutputVisitor& ar) const override;
        static void visitStatic(StaticVisitor&);
        void visit(StaticVisitor&) const override;

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
    void TDataContainer<T>::load(ILoadVisitor& visitor)
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainer<T>::save(ISaveVisitor& visitor) const
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainer<T>::visitStatic(StaticVisitor& visitor)
    {
        visitor.template visit<Header>("header");
        visitor.template visit<T>("data");
    }

    template <class T>
    void TDataContainer<T>::visit(StaticVisitor& visitor) const
    {
        visitStatic(visitor);
    }

    template <class T>
    void TDataContainer<T>::load(BinaryInputVisitor& ar)
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
    }

    template <class T>
    void TDataContainer<T>::save(BinaryOutputVisitor& ar) const
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

namespace cereal
{
    //! Saving for boost::optional
    template <class Archive, class Optioned>
    inline void save(Archive& ar, ::boost::optional<Optioned> const& optional)
    {
        bool init_flag(optional);
        if (init_flag)
        {
            ar(cereal::make_nvp("initialized", true));
            ar(cereal::make_nvp("value", optional.get()));
        }
        else
        {
            ar(cereal::make_nvp("initialized", false));
        }
    }

    //! Loading for boost::optional
    template <class Archive, class Optioned>
    inline void load(Archive& ar, ::boost::optional<Optioned>& optional)
    {

        bool init_flag;
        ar(cereal::make_nvp("initialized", init_flag));
        if (init_flag)
        {
            Optioned val;
            ar(cereal::make_nvp("value", val));
            optional = val;
        }
        else
        {
            optional = ::boost::none; // this is all we need to do to reset the internal flag and value
        }
    }
} // namespace cereal
