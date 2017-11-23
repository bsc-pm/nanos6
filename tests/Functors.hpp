/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef FUNCTORS_HPP
#define FUNCTORS_HPP


namespace Functors {
	struct Functor {};
	
	char functor_traits_helper(Functor *);
	long functor_traits_helper(...);
	
	template <typename T>
	struct is_functor {
		enum {
			value = (sizeof( functor_traits_helper( (T *) 0 ) ) == sizeof(char))
		};
	};
	
	
	template <typename T>
	class Ref {
	protected:
		T &_v;
		
	public:
		Ref(T &ref)
			: _v(ref)
		{
		}
		
		operator T & ()
		{
			return _v;
		}
		
		T &operator()()
		{
			return _v;
		}
	};
	
	
	template <typename T, bool isf = is_functor<T>::value>
	class RefOrFunctor : public Ref<T> {
	public:
		RefOrFunctor(T &ref)
			: Ref<T>(ref)
		{
		}
		
		RefOrFunctor(Ref<T> ref)
			: Ref<T>(ref)
		{
		}
	};
	
	
	template <typename T>
	class RefOrFunctor<T, true> {
		T _functor;
		
	public:
		RefOrFunctor(T functor)
			: _functor(functor)
		{
		}
		
		typename T::type operator()()
		{
			return _functor();
		}
	};
	
	
	template <typename T1, typename T2>
	class Base : public Functor {
	protected:
		RefOrFunctor<T1> _v1;
		RefOrFunctor<T2> _v2;
		
	public:
		Base(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: _v1(v1), _v2(v2)
		{
		}
	};
	
	template <typename T1, typename T2>
	class Equal : public Base<T1, T2> {
	public:
		typedef bool type;
		
		Equal(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1() == Base<T1, T2>::_v2();
		}
	};	
	
	template <typename T1, typename T2>
	class Different : public Base<T1, T2> {
	public:
		typedef bool type;
		
		Different(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1() != Base<T1, T2>::_v2();
		}
	};	
	
	template <typename T1, typename T2>
	class Greater : public Base<T1, T2> {
	public:
		typedef bool type;
		
		Greater(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1 > Base<T1, T2>::_v2;
		}
	};	
	
	template <typename T1, typename T2>
	class GreaterOrEqual : public Base<T1, T2> {
	public:
		typedef bool type;
		
		GreaterOrEqual(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1() >= Base<T1, T2>::_v2();
		}
	};	
	
	template <typename T1, typename T2>
	class Less : public Base<T1, T2> {
	public:
		typedef bool type;
		
		Less(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1() < Base<T1, T2>::_v2();
		}
	};	
	
	template <typename T1, typename T2>
	class LessOrEqual : public Base<T1, T2> {
	public:
		typedef bool type;
		
		LessOrEqual(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return Base<T1, T2>::_v1() <= Base<T1, T2>::_v2();
		}
	};
	
	template <typename T>
	class True : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		True(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return _v();
		}
	};
	
	template <typename T>
	class False : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		False(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return ! (bool) _v();
		}
	};
	
	template <typename T1, typename T2>
	class And : public Base<T1, T2> {
	public:
		typedef bool type;
		
		And(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return ((bool) Base<T1, T2>::_v1()) && ((bool) Base<T1, T2>::_v2());
		}
	};
	
	template <typename T1, typename T2>
	class Or : public Base<T1, T2> {
	public:
		typedef bool type;
		
		Or(RefOrFunctor<T1> v1, RefOrFunctor<T2> v2)
			: Base<T1, T2>(v1, v2)
		{
		}
		
		bool operator()()
		{
			return ((bool) Base<T1, T2>::_v1()) && ((bool) Base<T1, T2>::_v2());
		}
	};
	
	template <typename T>
	class Zero : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		Zero(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return _v() == 0;
		}
	};
	
	template <typename T>
	class NotZero : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		NotZero(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return _v() != 0;
		}
	};
	
	template <typename T>
	class One : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		One(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return _v() == 1;
		}
	};
	
	template <typename T>
	class NotOne : public Functor {
		RefOrFunctor<T> _v;
		
	public:
		typedef bool type;
		
		NotOne(RefOrFunctor<T> v)
			: _v(v)
		{
		}
		
		bool operator()()
		{
			return _v() != 1;
		}
	};
	
}

#endif // FUNCTORS_HPP
