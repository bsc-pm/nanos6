
EXTRA_DIST = centos/nanos6.spec \
	     debian/compat \
	     debian/source/format \
	     debian/rules \
	     debian/control \
	     Dockerfile.centos \
	     Dockerfile.debian \
	     README.md \
	     $(END)


rpm:
	@echo "Generating RPM structure"
	mkdir -p SOURCES BUILD RPMS SPECS SRPMS
	cp $(top_build_prefix)*.tar.bz2 SOURCES
	RELEASE=$${RELEASE:-@RELEASE_DATE@}; \
	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1}; \
	export MAKEFLAGS+=" -j $${THREADS}"; \
	rpmbuild -v -bb --clean $(srcdir)/centos/nanos6.spec \
		--define '_topdir $(CURDIR)' \
		--define '_version $(VERSION)' \
		--define "_release $${RELEASE}" \
		--define "_with_boost $${BOOST}"

clean-rpm:
	rm -rf BUILDROOT SOURCES BUILD RPMS SPECS SRPMS


deb:
	cp $(top_build_prefix)*.tar.bz2 nanos6_$(VERSION).orig.tar.bz2
	tar -xf nanos6_$(VERSION).orig.tar.bz2
	rm -rf upstream_dir
	mv nanos6-$(VERSION) upstream_dir/
	cp -rf $(srcdir)/debian upstream_dir/
	cp upstream_dir/COPYING upstream_dir/debian/copyright
	rm -f upstream_dir/debian/changelog
	RELEASE=$${RELEASE:-@RELEASE_DATE@}@DEB_RELEASE@; \
	if test ! -d $(top_srcdir)/.git; \
	then dch --create -c upstream_dir/debian/changelog --empty --package nanos6 -v "$(VERSION)-$${RELEASE}"; \
	else git --git-dir="$(top_srcdir)/.git" --work-tree="$(top_srcdir)/.git" log -1 --pretty=format:"nanos6 ($(VERSION)-$${RELEASE}) unstable; urgency=low%x0A%x0A  * %h %s%x0A%x0A -- %an <%ae>  %aD%n" > upstream_dir/debian/changelog; \
	fi
	cd upstream_dir; \
	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1}; \
	DEB_BUILD_OPTIONS="noddebs nocheck parallel=$${THREADS}" \
	debuild --preserve-envvar=BOOST -us -uc
	rm -rf upstream_dir

clean-deb:
	rm -f *.tar.bz2 *.dsc *.build *.changes *.deb
	rm -rf upstream_dir


clean-local: clean-rpm clean-deb
