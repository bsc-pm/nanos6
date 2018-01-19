#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([SSS_CHECK_CC_VERSION],
	[
		AC_MSG_CHECKING([the ${CC} version])
		if test x"$CC" != x"" ; then
			CC_VERSION=$(${CC} --version | head -1)
		fi
		AC_MSG_RESULT([$CC_VERSION])
		AC_SUBST([CC_VERSION])
	]
)


AC_DEFUN([SSS_CHECK_CXX_VERSION],
	[
		AC_MSG_CHECKING([the ${CXX} version])
		if test x"$CC" != x"" ; then
			CXX_VERSION=$(${CXX} --version | head -1)
		fi
		AC_MSG_RESULT([$CXX_VERSION])
		AC_SUBST([CXX_VERSION])
	]
)


AC_DEFUN([SSS_CHECK_SOURCE_VERSION],
	[
		AC_ARG_WITH(
			[git],
			[AS_HELP_STRING([--with-git=prefix], [specify the installation prefix of the git content tracker])],
			[ac_use_git_prefix="${withval}"],
			[ac_use_git_prefix=""]
		)
		
		if test x"${ac_use_git_prefix}" != x"" ; then
			AC_PATH_PROG([GIT], [git], [], [${ac_use_git_prefix}/bin])
		else
			AC_PATH_PROG([GIT], [git], [])
		fi
		AC_MSG_CHECKING([the source code version])
		if test -d "${srcdir}/$3/.git" -o -f "${srcdir}/$3/.git" ; then
			if test x"${GIT}" = x"" ; then
				AC_MSG_ERROR([need git to retrieve the source version information. Check the --with-git parameter.])
			fi
			SOURCE_VERSION=$($GIT --git-dir="${srcdir}/$3/.git" show --pretty=format:'%ci %h' -s HEAD)
			SOURCE_BRANCH=$($GIT --git-dir="${srcdir}/$3/.git" symbolic-ref HEAD | sed 's@refs/heads/@@')
			ac_source_in_git=true
		else
			if test x"$1" = x""; then
				SOURCE_VERSION=unknown
				SOURCE_BRANCH=unknown
			else
				SOURCE_VERSION="$1"
				SOURCE_BRANCH="$2"
			fi
			ac_source_in_git=false
		fi
		AC_MSG_RESULT([$SOURCE_BRANCH $SOURCE_VERSION])
		AC_SUBST([SOURCE_VERSION])
		AC_SUBST([SOURCE_BRANCH])
		
		AM_CONDITIONAL([FROM_GIT_REPOSITORY], [test x"${ac_source_in_git}" = x"true"])
		
		RELEASE_DATE=$(date +%Y%m%d)
		AC_CHECK_PROG([DEB_RELEASE], [lsb_release], [$(lsb_release -sc)], [])
		AC_SUBST([RELEASE_DATE])
		AC_SUBST([DEB_RELEASE])
	]
)

