#include <../tests/reflect/Data.hpp>
#include <../tests/reflect/Reflect.hpp>
#include <../tests/reflect/common.hpp>

#include <MetaObject/serialization/BinaryReader.hpp>

#include <MetaObject/serialization/BinaryWriter.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <MetaObject/params/visitor_traits/map.hpp>
#include <MetaObject/params/visitor_traits/memory.hpp>
#include <MetaObject/params/visitor_traits/string.hpp>
#include <MetaObject/params/visitor_traits/vector.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/cerealize.hpp>
#include <ct/reflect/compare-inl.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print.hpp>

#include <ct/reflect/compare-container-inl.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <cassert>
#include <fstream>
#include <map>
#include <type_traits>

namespace cereal
{
    //! Saving for std::map<std::string, std::string> for text based archives
    // Note that this shows off some internal cereal traits such as EnableIf,
    // which will only allow this template to be instantiated if its predicates
    // are true
    template <class Archive,
              class T,
              class C,
              class A,
              traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void save(Archive& ar, std::map<std::string, T, C, A> const& map)
    {
        for (const auto& i : map)
            ar(cereal::make_nvp(i.first, i.second));
    }

    //! Loading for std::map<std::string, std::string> for text based archives
    template <class Archive,
              class T,
              class C,
              class A,
              traits::EnableIf<traits::is_text_archive<Archive>::value> = traits::sfinae>
    inline void load(Archive& ar, std::map<std::string, T, C, A>& map)
    {
        map.clear();

        auto hint = map.begin();
        while (true)
        {
            const auto namePtr = ar.getNodeName();

            if (!namePtr)
                break;

            std::string key = namePtr;
            T value;
            ar(value);
            hint = map.emplace_hint(hint, std::move(key), std::move(value));
        }
    }
} // namespace cereal

struct Vec
{
    float x, y, z;
};

namespace mo
{
    template <>
    struct TTraits<Vec, void> : IStructTraits
    {
        Vec* m_vec;
        using base = IStructTraits;
        TTraits(Vec* vec)
            : m_vec(vec)
        {
        }

        virtual size_t size() const override
        {
            return sizeof(Vec);
        }
        virtual bool isPrimitiveType() const override
        {
            return false;
        }
        virtual bool triviallySerializable() const override
        {
            return std::is_pod<Vec>::value;
        }
        virtual void visit(IReadVisitor* visitor) override
        {
            (*visitor) (&m_vec->x, "x") (&m_vec->y, "y")(&m_vec->z, "z");
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            (*visitor) (&m_vec->x, "x") (&m_vec->y, "y")(&m_vec->z, "z");
        }
        virtual const void* ptr() const override
        {
            return m_vec;
        }
        virtual void* ptr() override
        {
            return m_vec;
        }
    };

    template <class T1, class T2>
    struct TTraits<std::pair<T1, T2>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(std::pair<T1, T2>* ptr, const std::pair<T1, T2>* const_ptr)
            : m_ptr(ptr)
            , m_const_ptr(const_ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->first, "first");
            (*visitor)(&m_ptr->second, "second");
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                (*visitor)(&m_const_ptr->first, "first");
                (*visitor)(&m_const_ptr->second, "second");
            }
            else
            {
                (*visitor)(&m_ptr->first, "first");
                (*visitor)(&m_ptr->second, "second");
            }
        }
        virtual size_t size() const
        {
            return sizeof(std::pair<T1, T2>);
        }
        virtual bool triviallySerializable() const
        {
            return std::is_pod<T1>::value && std::is_pod<T2>::value;
        }
        virtual bool isPrimitiveType() const
        {
            return false;
        }
        virtual const void* ptr() const
        {
            return m_ptr ? m_ptr : m_const_ptr;
        }
        virtual void* ptr()
        {
            return m_ptr;
        }

      private:
        std::pair<T1, T2>* m_ptr;
        const std::pair<T1, T2>* m_const_ptr;
    };
}

struct TestBinary
{
    template <class T>
    void test(const T& data)
    {
        {
            std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
            mo::BinaryWriter bar(ofs);
            mo::IWriteVisitor& visitor = bar;
            T tmp = data;
            visitor(&tmp, "value0");
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryReader bar(ifs);
            mo::IReadVisitor& visitor = bar;

            T tmp;
            visitor(&tmp, "value0");
            if (!ct::compare(tmp, data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
        }
    }

    template <class T>
    void test(const std::shared_ptr<T>& data)
    {
        {
            std::ofstream ofs("test.bin", std::ios::binary | std::ios::out);
            mo::BinaryWriter bar(ofs);
            mo::IWriteVisitor& visitor = bar;
            std::shared_ptr<T> tmp = data;
            visitor(&tmp, "value0");
            visitor(&tmp, "value1");
        }
        {
            std::ifstream ifs("test.bin", std::ios::binary | std::ios::in);
            mo::BinaryReader bar(ifs);
            mo::IReadVisitor& visitor = bar;

            std::shared_ptr<T> tmp;
            std::shared_ptr<T> tmp2;
            visitor(&tmp, "value0");
            visitor(&tmp2, "value1");
            if (!ct::compare(tmp, data, DebugEqual()))
            {
                std::cout << "Failed to serialize " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Serialization failed");
            }
            if (tmp.get() != tmp2.get())
            {
                std::cout << "Failed to reshare ownership of two smart pointers" << std::endl;
                throw std::runtime_error("Serialization failed");
            }
        }
    }
};

struct TestJson
{
    template <class T>
    void test(const T& data)
    {
        T tmp = data;
        std::stringstream ss;
        {
            mo::JSONWriter printer(ss);
            mo::IWriteVisitor& visitor = printer;
            visitor(&tmp);
        }
        ss.seekg(std::ios::beg);
        std::cout << "------------------------------\nDynamic\n";
        std::cout << ss.str() << std::endl;
        std::cout << "------------------------------\nStatic\n";
        {
            cereal::JSONOutputArchive ar(std::cout);
            ar(tmp);
        }

        {
            cereal::JSONInputArchive ar(ss);
            T tmp1;
            ar(tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Static Json deserialization failed");
            }
        }

        {
            std::stringstream ss1(ss.str());
            // TODO dynamic json deserialization
            mo::JSONReader reader(ss1);
            mo::IReadVisitor& visitor = reader;
            T tmp1;
            visitor(&tmp1);
            if (!ct::compare(tmp, tmp1, DebugEqual()))
            {
                std::cout << "Failed to load from json " << ct::Reflect<T>::getName() << " correctly";
                throw std::runtime_error("Dynamic Json deserialization failed");
            }
        }
    }

    template <class T>
    void test(const std::shared_ptr<T>& data)
    {
        std::shared_ptr<T> tmp = data;
        std::stringstream ss;
        {
            mo::JSONWriter printer(ss);
            mo::IWriteVisitor& visitor = printer;
            visitor(&tmp);
        }
        ss.seekg(std::ios::beg);
        std::cout << "------------------------------\nDynamic\n";
        std::cout << ss.str() << std::endl;
        std::cout << "------------------------------\nStatic\n";
        {
            cereal::JSONOutputArchive ar(std::cout);
            ar(tmp);
        }
    }
};

int main()
{
    {
        using Accessor_t = decltype(ct::Reflect<WeirdWeakOwnerShip>::getAccessor(ct::Indexer<1>{}));
        using Get_t = Accessor_t::RetType;
        static_assert(std::is_same<const std::vector<PointerOwner>&, Get_t>::value, "asdf");
    }
    TestBinary tester;
    testTypes(tester);

    TestJson test_json;
    testTypes(test_json);

    std::cout << std::endl;
    std::shared_ptr<ReflectedStruct> shared_ptr = std::make_shared<ReflectedStruct>();

    {
        mo::JSONWriter writer(std::cout);
        mo::IWriteVisitor& visitor = writer;
        visitor(&shared_ptr);
        visitor(&shared_ptr);
    }
    std::cout << std::endl;
}
