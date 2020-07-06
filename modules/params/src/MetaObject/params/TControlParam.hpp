#ifndef MO_PARAMS_TCONTROLPARAM_HPP
#define MO_PARAMS_TCONTROLPARAM_HPP
#include <MetaObject/params/ITControlParam.hpp>
namespace mo
{
    template <class T>
    struct TControlParam : ITControlParam<T>
    {
        TControlParam() = default;
        T& getValue() override;
        const T& getValue() const override;
        void setValue(T&& val) override;

        std::ostream& print(std::ostream&) const;

        bool getModified() const;
        void setModified(bool val);

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void visit(StaticVisitor&) const override;

      private:
        bool m_modified = false;
        T m_data;
    };

    template <class T>
    struct TControlParam<T*> : ITControlParam<T>
    {
        TControlParam() = default;
        T& getValue() override;
        const T& getValue() const override;
        void setValue(T&& val) override;

        void setUserDataPtr(T*);

        std::ostream& print(std::ostream&) const;

        bool getModified() const;
        void setModified(bool val);

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void visit(StaticVisitor&) const override;

      private:
        bool m_modified = false;
        T* m_ptr = nullptr;
    };

    template <class T>
    struct TStateParam : TControlParam<T*>
    {
        TStateParam()
        {
            this->setFlags(ParamFlags::kSTATE);
        }
    };

    /////////////////////////////////////////////////////////////
    // TControlParam<T>
    /////////////////////////////////////////////////////////////
    template <class T>
    T& TControlParam<T>::getValue()
    {
        return m_data;
    }

    template <class T>
    const T& TControlParam<T>::getValue() const
    {
        return m_data;
    }

    template <class T>
    void TControlParam<T>::setValue(T&& val)
    {
        if (&val != &m_data)
        {
            m_data = std::move(val);
        }
        this->setModified(true);
    }

    template <class T>
    std::ostream& TControlParam<T>::print(std::ostream& os) const
    {
        os << '[' << this->getTreeName() << "] " << this->getTypeInfo() << " = " << m_data;
        return os;
    }

    template <class T>
    bool TControlParam<T>::getModified() const
    {
        return m_modified;
    }

    template <class T>
    void TControlParam<T>::setModified(bool val)
    {
        m_modified = val;
    }

    template <class T>
    void TControlParam<T>::load(ILoadVisitor& visitor)
    {
        ITControlParam<T>::load(visitor);
        visitor(&m_data, "data");
    }
    template <class T>
    void TControlParam<T>::save(ISaveVisitor& visitor) const
    {
        ITControlParam<T>::save(visitor);
        visitor(&m_data, "data");
    }

    template <class T>
    void TControlParam<T>::visit(StaticVisitor& visitor) const
    {
        ITControlParam<T>::visit(visitor);
        visitor.template visit<T>("data");
    }

    /////////////////////////////////////////////////////////////
    // TControlParam<T*>
    /////////////////////////////////////////////////////////////

    template <class T>
    T& TControlParam<T*>::getValue()
    {
        MO_ASSERT_LOGGER(this->getLogger(), m_ptr != nullptr);
        return *m_ptr;
    }

    template <class T>
    const T& TControlParam<T*>::getValue() const
    {
        MO_ASSERT_LOGGER(this->getLogger(), m_ptr != nullptr);
        return *m_ptr;
    }

    template <class T>
    void TControlParam<T*>::setValue(T&& val)
    {
        MO_ASSERT_LOGGER(this->getLogger(), m_ptr != nullptr);
        if (&val != m_ptr)
        {
            *m_ptr = std::move(val);
        }
        this->setModified(true);
    }

    template <class T>
    void TControlParam<T*>::setUserDataPtr(T* ptr)
    {
        m_ptr = ptr;
    }

    template <class T>
    std::ostream& TControlParam<T*>::print(std::ostream& os) const
    {
        os << '[' << this->getTreeName() << "] " << this->getTypeInfo();
        if (m_ptr)
        {
            os << " = " << *m_ptr;
        }
        else
        {
            os << " un-initialized";
        }
        return os;
    }

    template <class T>
    bool TControlParam<T*>::getModified() const
    {
        return m_modified;
    }

    template <class T>
    void TControlParam<T*>::setModified(bool val)
    {
        m_modified = val;
    }

    template <class T>
    void TControlParam<T*>::load(ILoadVisitor& visitor)
    {
        ITControlParam<T>::load(visitor);
        if (m_ptr)
        {
            visitor(m_ptr, "data");
        }
    }

    template <class T>
    void TControlParam<T*>::save(ISaveVisitor& visitor) const
    {
        ITControlParam<T>::save(visitor);
        if (m_ptr)
        {
            visitor(m_ptr, "data");
        }
    }

    template <class T>
    void TControlParam<T*>::visit(StaticVisitor& visitor) const
    {
        ITControlParam<T>::visit(visitor);
        visitor.template visit<T>("data");
    }
} // namespace mo

#endif // MO_PARAMS_TCONTROLPARAM_HPP