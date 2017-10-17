
# Disable debuginfo package
%define debug_package %{nil}

Name:           nanos6
Version:        %{_version}
Release:        %{_release}%{?dist}
Summary:        Nanos6 summary

Group:          Development/Tools
License:        Nanos6 license
URL:            http://www.bsc.es

Prefix:         %{_prefix}
Source:         nanos6-%{_version}.tar.bz2
BuildRequires:  numactl-devel hwloc-devel papi-devel elfutils-devel
Requires:       numactl-libs hwloc-libs papi elfutils

%description
Nanos6 description


%prep
%setup -q


%build
%configure --without-nanos6-mercurium --with-boost=%{_with_boost}
make %{?_smp_mflags}


%install
make install DESTDIR=%{buildroot}


%files
%{_libdir}/*
%{_includedir}/*
%{_datarootdir}/doc/nanos6/*
