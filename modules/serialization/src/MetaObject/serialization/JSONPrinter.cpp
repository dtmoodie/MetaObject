#include "JSONPrinter.hpp"
#include <MetaObject/logging/logging.hpp>
#include <cereal/archives/json.hpp>
#include <ct/types/TArrayView.hpp>

namespace mo
{
    struct JSONSaver::Impl
    {
        Impl(std::ostream& stream)
            : m_ar(stream)
        {
        }

        cereal::JSONOutputArchive m_ar;
    };

    JSONSaver::JSONSaver(std::ostream& os)
        : m_impl(std::make_unique<Impl>(os))
    {
    }

    JSONSaver::~JSONSaver() = default;

    template <class T>
    ISaveVisitor& JSONSaver::writePod(const T* ptr, const std::string& name, const size_t cnt)
    {
        if (cnt == 0)
        {
            return *this;
        }
        if (!name.empty())
        {
            m_impl->m_ar.setNextName(name.c_str());
        }
        if (cnt > 1)
        {
            // m_ar.startNode();
            // m_ar.makeArray();
            for (size_t i = 0; i < cnt; ++i)
            {
                m_impl->m_ar(ptr[i]);
            }
            // m_ar.finishNode();
        }
        if (cnt == 1)
        {
            m_impl->m_ar(ptr[0]);
        }

        return *this;
    }

    ISaveVisitor& JSONSaver::operator()(const bool* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const char* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const int8_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const uint8_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const int16_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const uint16_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const int32_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const uint32_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const int64_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const uint64_t* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
    ISaveVisitor& JSONSaver::operator()(const long long* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const unsigned long long* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }
#endif
#else
    ISaveVisitor& JSONSaver::operator()(const long int* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const unsigned long int* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }
#endif
    ISaveVisitor& JSONSaver::operator()(const float* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const double* val, const std::string& name, const size_t cnt)
    {
        return writePod(val, name, cnt);
    }

    ISaveVisitor& JSONSaver::operator()(const void* val, const std::string& name, const size_t cnt)
    {
        const char* name_ptr = nullptr;
        if (!name.empty())
        {
            name_ptr = name.c_str();
        }

        m_impl->m_ar.saveBinaryValue(val, cnt, name_ptr);
        return *this;
    }

    ISaveVisitor& JSONSaver::operator()(const IStructTraits* val, const void* inst, const std::string& name, size_t cnt)
    {
        if (!name.empty())
        {
            m_impl->m_ar.setNextName(name.c_str());
        }
        const uint8_t* ptr = ct::ptrCast<const uint8_t>(inst);
        const uint32_t num_members = val->getNumMembers();
        for (size_t i = 0; i < cnt; ++i)
        {
            m_impl->m_ar.startNode();
            auto tptr = ct::ptrCast<const void>(ptr);
            bool member_save_success = false;
            const bool is_ptr = val->isPtr();
            if (num_members == 1 && !is_ptr)
            {
                member_save_success = val->saveMember(*this, tptr, 0);
            }
            if (!member_save_success)
            {
                SaveCache::operator()(val, tptr, name, 1);
            }

            ptr += val->size();
            m_impl->m_ar.finishNode();
        }
        return *this;
    }

    ISaveVisitor&
    JSONSaver::operator()(const IContainerTraits* val, const void* inst_, const std::string& name, size_t cnt)
    {
        auto str_type = TypeInfo::create<std::string>();
        const bool saving_string = val->type() == str_type;
        const bool saving_binary = val->valueType() == TypeInfo::Void();
        const bool saving_string_dictionary = val->keyType() == str_type;

        const uint8_t* inst = ct::ptrCast<uint8_t>(inst_);

        for (size_t i = 0; i < cnt; ++i)
        {
            if (!name.empty())
            {
                m_impl->m_ar.setNextName(name.c_str());
            }

            if (saving_string)
            {
                m_impl->m_ar.writeName();
                m_impl->m_ar.saveValue(*ct::ptrCast<const std::string>(inst));
                inst += val->size();
                continue;
            }

            if (saving_binary)
            {
                // const uint64_t size = val->getContainerSize(inst);
                // m_ar(cereal::make_nvp("size", size));
            }

            if (!saving_binary)
            {
                m_impl->m_ar.startNode();
                if (!saving_string_dictionary)
                {
                    m_impl->m_ar.makeArray();
                }
            }

            val->save(*this, inst, name, cnt);

            if (!saving_binary)
            {
                m_impl->m_ar.finishNode();
            }
            inst += val->size();
        }

        return *this;
    }

    VisitorTraits JSONSaver::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.human_readable = true;
        return out;
    }

    std::shared_ptr<Allocator> JSONSaver::getAllocator() const
    {
        return m_allocator;
    }

    void JSONSaver::setAllocator(std::shared_ptr<Allocator> alloc)
    {
        m_allocator = std::move(alloc);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                                      JSONLoader
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct JSONLoader::Impl
    {
        Impl(std::istream& os)
            : m_ar(os)
        {
        }
        cereal::JSONInputArchive m_ar;
    };

    JSONLoader::JSONLoader(std::istream& os)
        : m_impl(std::make_unique<Impl>(os))
    {
    }

    JSONLoader::~JSONLoader() = default;

    template <class T>
    ILoadVisitor& JSONLoader::readPod(T* ptr, const std::string& name, const size_t cnt)
    {
        /*if (cnt == 0)
        {
            return *this;
        }*/
        try
        {
            if (!name.empty())
            {
                m_impl->m_ar.setNextName(name.c_str());
            }
            auto node_name = m_impl->m_ar.getNodeName();
            if (cnt > 1)
            {
                // m_ar.startNode();
                cereal::size_type size;
                m_impl->m_ar.loadSize(size);
                MO_ASSERT_EQ(size, cnt);
            }
            for (size_t i = 0; i < cnt; ++i)
            {
                node_name = m_impl->m_ar.getNodeName();
                m_impl->m_ar(ptr[i]);
                if (node_name)
                {
                    m_last_read_name = name;
                }
            }
            if (cnt > 1)
            {
                // m_ar.finishNode();
            }
        }
        catch (cereal::Exception& e)
        {
            MO_LOG(info,
                   "Failure to deserialize {} from json due to {}\n{}",
                   name,
                   e.what(),
                   boost::stacktrace::stacktrace());
        }

        return *this;
    }

    ILoadVisitor& JSONLoader::operator()(bool* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(char* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(int8_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(uint8_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(int16_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(uint16_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(int32_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(uint32_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(int64_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(uint64_t* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
    ILoadVisitor& JSONLoader::operator()(long long* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(unsigned long long* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
#endif
#else
    ILoadVisitor& JSONLoader::operator()(long int* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(unsigned long int* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }
#endif
    ILoadVisitor& JSONLoader::operator()(float* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(double* val, const std::string& name, const size_t cnt)
    {
        return readPod(val, name, cnt);
    }

    ILoadVisitor& JSONLoader::operator()(void* val, const std::string& name, const size_t cnt)
    {
        const char* name_ptr = nullptr;
        if (!name.empty())
        {
            name_ptr = name.c_str();
        }
        m_impl->m_ar.loadBinaryValue(val, cnt, name_ptr);
        return *this;
    }

    ILoadVisitor& JSONLoader::operator()(const IStructTraits* val, void* inst, const std::string& name, size_t cnt)
    {
        if (!name.empty())
        {
            m_impl->m_ar.setNextName(name.c_str());
        }
        auto ptr = ct::ptrCast<uint8_t>(inst);
        const uint32_t num_members = val->getNumMembers();
        for (size_t i = 0; i < cnt; ++i)
        {
            auto name_ptr = m_impl->m_ar.getNodeName();
            m_impl->m_ar.startNode();
            auto tptr = ct::ptrCast<void>(ptr);
            bool member_load_success = false;
            const bool is_ptr = val->isPtr();
            if (num_members == 1 && !is_ptr)
            {
                member_load_success = val->loadMember(*this, tptr, 0);
            }
            if (!member_load_success)
            {
                LoadCache::operator()(val, tptr, name, 1);
            }

            if (name_ptr)
            {
                m_last_read_name = name_ptr;
            }
            ptr += val->size();
            m_impl->m_ar.finishNode();
        }

        return *this;
    }

    ILoadVisitor& JSONLoader::operator()(const IContainerTraits* val, void* inst_, const std::string& name, size_t cnt)
    {
        static const TypeInfo string_type = TypeInfo::create<std::string>();
        const bool loading_string = val->type() == string_type;
        const bool loading_binary =
            val->valueType() == TypeInfo::Void() || val->valueType() == TypeInfo::create<Byte>();
        const bool loading_string_dict = val->keyType() == string_type;
        uint8_t* inst = ct::ptrCast<uint8_t>(inst_);
        for (size_t i = 0; i < cnt; ++i)
        {
            std::string container_name = name;
            const char* node_name = m_impl->m_ar.getNodeName();
            const bool already_in_node = node_name != nullptr ? container_name == node_name : false;
            if (loading_string)
            {
                m_impl->m_ar.loadValue(*ct::ptrCast<std::string>(inst));
                inst += val->size();
                continue;
            }

            if (!name.empty() && !already_in_node)
            {
                m_impl->m_ar.setNextName(name.c_str());
            }

            if (!loading_string)
            {
                if (node_name && !loading_binary)
                {
                    m_impl->m_ar.startNode();
                }

                if (node_name && name == node_name)
                {
                    container_name.clear();
                }

                if (loading_binary)
                {
                    // uint64_t size = 0;
                    // m_ar(cereal::make_nvp("size", size));
                    // val->setContainerSize(size, inst);
                    // m_current_size = size;
                }
                else
                {
                    if (!loading_string_dict)
                    {
                        uint64_t size;
                        m_impl->m_ar.loadSize(size);
                        val->setContainerSize(size, inst);
                        m_current_size = size;
                    }
                    else
                    {
                        uint64_t count = 0;
                        while (true)
                        {
                            const auto name = m_impl->m_ar.getNodeName();
                            if (!name)
                            {
                                break;
                            }
                            count += 1;
                            m_impl->m_ar.startNode();
                            m_impl->m_ar.finishNode();
                        }
                        val->setContainerSize(count, inst);
                        m_current_size = count;
                        m_impl->m_ar.finishNode();
                        if (node_name)
                        {
                            m_impl->m_ar.setNextName(node_name);
                        }

                        m_impl->m_ar.startNode();
                    }
                }
            }
            val->load(*this, inst, container_name, cnt);
            if (!loading_string && !loading_binary)
            {
                m_impl->m_ar.finishNode();
            }
            inst += val->size();
        }

        return *this;
    }

    VisitorTraits JSONLoader::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.human_readable = true;
        return out;
    }

    std::string JSONLoader::getCurrentElementName() const
    {
        return m_last_read_name;
    }

    size_t JSONLoader::getCurrentContainerSize() const
    {
        return m_current_size;
    }

    std::shared_ptr<Allocator> JSONLoader::getAllocator() const
    {
        return m_allocator;
    }

    void JSONLoader::setAllocator(std::shared_ptr<Allocator> alloc)
    {
        m_allocator = std::move(alloc);
    }
} // namespace mo
