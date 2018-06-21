#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2010-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AX_CHECK_EXTRAE],[

AC_ARG_WITH(extrae,
   [AS_HELP_STRING([--with-extrae,--with-extrae=PATH], [search/specify home directory for the extrae package.])],
   [with_extrae=${withval}],
   [with_extrae=check])

   extrae_CPPFLAGS=
   extrae_LDFLAGS=
   extrae_LIBS=

   saved_LIBS=${LIBS}
   LIBS=

   have_extrae=no
   case "x${with_extrae}" in
      xno)
         ;;
      xyes)
         AC_MSG_NOTICE([will requiere Extrae to continue])
         AC_SEARCH_LIBS([Extrae_init], [nanostrace], [have_extrae=yes], [have_extrae=no])
         if test "x${have_extrae}" = "xno"; then
            AC_MSG_ERROR([Extrae library not found])
         fi
         AC_CHECK_HEADERS([extrae_types.h], [], [have_extrae=no])
         if test "x${have_extrae}" = "xno"; then
            AC_MSG_ERROR([Extrae headers not found])
         fi
         extrae_LIBS=${LIBS}
         ;;
      xcheck)
         AC_MSG_NOTICE([will use Extrae only if present])
         AC_SEARCH_LIBS([Extrae_init], [nanostrace], [have_extrae=yes], [have_extrae=no])
         AC_CHECK_HEADERS([extrae_types.h], [], [have_extrae=no])
         extrae_LIBS=${LIBS}
         ;;
      x)
         AC_MSG_ERROR([Check --with-extrae configuration flag usage])
         ;;
      *)
         AC_MSG_NOTICE([checking Extrae at ${with_extrae}/lib])

         saved_LDFLAGS=${LDFLAGS}
         LDFLAGS=-L${with_extrae}/lib
         AC_SEARCH_LIBS([Extrae_init], [nanostrace], [have_extrae=yes], [have_extrae=no])
      
         if test "x${have_extrae}" = "xyes"; then
            saved_CPPFLAGS=${CPPFLAGS}
            CPPFLAGS=-I${with_extrae}/include
            AC_CHECK_HEADERS([extrae_types.h], [], [have_extrae=no])
            if test "x${have_extrae}" = "xyes"; then
               extrae_CPPFLAGS=${CPPFLAGS}
               extrae_LDFLAGS=${LDFLAGS}
               extrae_LIBS=${LIBS}
            fi
            CPPFLAGS=${saved_CPPFLAGS}
         fi

         LDFLAGS=${saved_LDFLAGS}
         ;;
   esac

   LIBS=${saved_LIBS}

   if test "x${have_extrae}" = xyes; then
      AC_MSG_NOTICE([Extrae module preprocessors flags: ${extrae_CPPFLAGS}])
      AC_MSG_NOTICE([Extrae module linker flags: ${extrae_LDFLAGS} ${extrae_LIBS}])
      AC_DEFINE(HAVE_EXTRAE,,[define if the Extrae library is available])
   fi

   AC_SUBST(extrae_CPPFLAGS)
   AC_SUBST(extrae_LDFLAGS)
   AC_SUBST(extrae_LIBS)

   AM_CONDITIONAL([HAVE_EXTRAE], [test "x${have_extrae}" = xyes])

]) dnl AC_DEFUN
