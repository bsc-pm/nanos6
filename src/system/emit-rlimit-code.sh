#!/bin/sh
cat /usr/include/x86_64-linux-gnu/bits/resource.h | awk '
	/RLIMIT_NLIMITS/ { exit; }
	/BSD/ { $0=""; }
	/RLIMIT_OFILE/ { $0=""; }
	/enum / {
		begin_comment=0;
		end_comment=0;
		comment="";
		units="";
	}
	/#define[ \t]+RLIMIT_/ {
		print "#ifdef " $2;
		print "\taddResourceLimitReportEntry(" $2 ", \"" tolower($2) "\", \"" comment "\", \"" units "\");";
		print "#endif";
		begin_comment=0;
		end_comment=0;
		comment="";
		units="";
	}
	/\/\*/ { begin_comment=1; }
	/./ {
		if (begin_comment == 1 && end_comment != 1) {
			for (i=1; i<=NF; i++) {
				if ($i != "/*" && $i != "*/") {
					if (comment == "") {
						comment = $i;
					} else {
						comment = comment " " $i;
					}
				}
			}
		}
	}
	/\*\// { end_comment=1; }
	/in seconds/ { units="seconds"; }
	/in bytes/ { units="bytes"; }
	/of processes/ { units="processes"; }
	/file locks/ { units="file locks"; }
	/pending signals/ { units="signals"; }
	/mum bytes / { units="bytes"; }
	/time in µs/ { units="µs"; }
	/[Aa]dress space/ { units="bytes"; }
' | sed 's/[.]"/"/'
