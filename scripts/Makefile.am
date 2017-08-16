
EXTRA_DIST = centos/nanos6.spec \
	     Dockerfile.centos \
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

clean-local: clean-rpm
