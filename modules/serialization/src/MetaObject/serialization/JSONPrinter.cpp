#include "JSONPrinter.hpp"
#include <MetaObject/logging/logging.hpp>
#include <ct/types/TArrayView.hpp>

namespace mo
{
    JSONSaver::JSONSaver(std::ostream& os)
        : m_ar(os)
    {
    }

    template <class T>
    ISaveVisitor& JSONSaver::writePod(const T* ptr, const std::string& name, const size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        if (cnt > 1)
        {
            m_ar.startNode();
            m_ar.makeArray();
            for (size_t i = 0; i < cnt; ++i)
            {
                m_ar(ptr[i]);
            }
            m_ar.finishNode();
        }
        else
        {
            m_ar(ptr[0]);
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
        m_ar.saveBinaryValue(val, cnt, name.c_str());
        return *this;
    }

    ISaveVisitor& JSONSaver::operator()(IStructTraits* val, const void* inst, const std::string& name, size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        const uint8_t* ptr = ct::ptrCast<const uint8_t>(inst);
        for (size_t i = 0; i < cnt; ++i)
        {
            m_ar.startNode();
            SaveCache::operator()(val, ct::ptrCast<const void>(ptr), name, 1);
            ptr += val->size();
            m_ar.finishNode();
        }
        return *this;
    }

    ISaveVisitor& JSONSaver::operator()(IContainerTraits* val, const void* inst, const std::string& name, size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.startNode();
            if (val->keyType() != TypeInfo(typeid(std::string)))
            {
                m_ar.makeArray();
            }
        }
        if (val->type() == TypeInfo::create<std::string>())
        {
            m_ar.writeName();
            m_ar.saveValue(*static_cast<const std::string*>(inst));
            return *this;
        }
        val->save(*this, inst, name, cnt);
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.finishNode();
        }
        return *this;
    }

    VisitorTraits JSONSaver::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.reader = false;
        return out;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///                                      JSONLoader
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    JSONLoader::JSONLoader(std::istream& os)
        : m_ar(os)
    {
    }

    template <class T>
    ILoadVisitor& JSONLoader::readPod(T* ptr, const std::string& name, const size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        auto node_name = m_ar.getNodeName();
        if (cnt > 1)
        {
            m_ar.startNode();
            cereal::size_type size;
            m_ar.loadSize(size);
            MO_ASSERT_EQ(size, cnt);
        }
        for (size_t i = 0; i < cnt; ++i)
        {
            node_name = m_ar.getNodeName();
            m_ar(ptr[i]);
            if (node_name)
            {
                m_last_read_name = name;
            }
        }
        if (cnt > 1)
        {
            m_ar.finishNode();
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
        m_ar.loadBinaryValue(val, cnt, name.c_str());
        return *this;
    }

    ILoadVisitor& JSONLoader::operator()(IStructTraits* val, void* inst, const std::string& name, size_t cnt)
    {
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        auto ptr = ct::ptrCast<uint8_t>(inst);
        for (size_t i = 0; i < cnt; ++i)
        {
            auto name_ptr = m_ar.getNodeName();
            m_ar.startNode();

            LoadCache::operator()(val, ct::ptrCast<void>(ptr), name, 1);
            if (name_ptr)
            {
                m_last_read_name = name_ptr;
            }
            ptr += val->size();
            m_ar.finishNode();
        }

        return *this;
    }

    ILoadVisitor& JSONLoader::operator()(IContainerTraits* val, void* inst, const std::string& name, size_t cnt)
    {
        MO_ASSERT_EQ(cnt, 1);
        std::string container_name = name;
        const std::string node_name = m_ar.getNodeName();
        if (val->type() == TypeInfo::create<std::string>() && name == node_name)
        {
            std::string val;
            m_ar.loadValue(val);
            *static_cast<std::string*>(inst) = val;
            return *this;
        }
        if (!name.empty())
        {
            m_ar.setNextName(name.c_str());
        }
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.startNode();
            if (node_name == name)
            {
                container_name.clear();
            }

            if (val->keyType() != TypeInfo(typeid(std::string)))
            {
                uint64_t size;
                m_ar.loadSize(size);
                val->setContainerSize(size, inst);
                m_current_size = size;
            }
            else
            {
                uint64_t count = 0;
                while (true)
                {
                    const auto name = m_ar.getNodeName();
                    if (!name)
                    {
                        break;
                    }
                    count += 1;
                    m_ar.startNode();
                    m_ar.finishNode();
                }
                val->setContainerSize(count, inst);
                m_current_size = count;
                m_ar.finishNode();
                m_ar.setNextName(node_name.c_str());
                m_ar.startNode();
            }
        }
        val->load(*this, inst, container_name, cnt);
        if (val->type() != TypeInfo(typeid(std::string)))
        {
            m_ar.finishNode();
        }
        return *this;
    }

    VisitorTraits JSONLoader::traits() const
    {
        VisitorTraits out;
        out.supports_named_access = true;
        out.reader = true;
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
} // namespace mo
