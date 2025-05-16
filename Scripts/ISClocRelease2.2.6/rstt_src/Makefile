#
# iscloc_rstt Makefile
#
# Istvan Bondar, ISC, September 2012 - November 2013
#

#
#
# Path for iscloc source files (edit this line!)
#
#
ISCPATH = ../src


objs = main_rstt.o phaseids_parallel_rstt.o traveltimes_parallel_rstt.o

iscloc_files = calc_ellipticity_corr.o \
               calc_uncertainty.o \
               cluster.o \
               data_covariance.o \
               depth_phases.o \
               distaz.o \
               gregion.o \
               initializations.o \
               interpolate.o \
               isf.o \
               locator_parallel.o \
               loc_qual.o \
               magnitude.o \
               na_parallel.o \
               pgsql_funcs.o \
               print_event.o \
               read_db.o \
               read_textfiles.o \
               svd_parallel.o \
               timefuncs.o \
               utils.o \
               write_db.o

iscloc_objs = $(iscloc_files:%=$(ISCPATH)/%)

#
#
# default compiler options
#    (on Mac OS gcc is a symbolic link to llvm-gcc)
#
CC = gcc
DEBUG = -Wall -g -O0

#
#
# Dependencies
#
#
#   LAPACK and LBLAS libraries
#    - MacOS X 10.6 and later: part of MacOS X Accelerate framework
#    - linux/unix: libraries are assumed to be in $(LAPACK)
#   PostgreSQL client 
#    - MacOS X 10.6 and later: part of MacOS X
#      Mac OS X 10.9 and later: PostgreSQL client was removed from the OS
#    - linux/unix: assumed that PostgreSL client is installed
#                  PostgreSQL include files are assumed to be in $(IPGSQL)
#   Grand Central Dispatch
#    - SLBM is not thread safe, cannot use Grand Central Dispatch 
#      for multicore processing
#   RSTT libraries
#    - RSTT files are assumed to be in $(SLBMROOT)
#
RSTTLIB = -L$(SLBMROOT)/lib -lslbm -lslbmCshell
RSTTINC = -I$(SLBMROOT)/SLBM/include -I$(SLBMROOT)/SLBM_C_shell/include

#
#
# Determine OS and set OS-specific compiler variables
#
#
OS := $(shell uname -s)
ifeq ($(OS),Darwin)
#
# Mac OS X 10.6 and later:
#    - Grand Central Dispatch and PostgreSQL client are part of the OS
#    - LAPACK is part of the Accelerate framework
#
	OS = darwin
	LAPACK = -framework Accelerate
	MACOS = -DMACOSX=1
# Mac OS X 10.9 and later:
#    - PostgreSQL client was removed from the OS
	IPGSQL = -I../postgresql_include
else
#
# Linux (tested on OpenSUSE, Redhat and Ubuntu):
#    - assumes that PostgreSL client is installed
#      PostgreSQL include files are assumed to be in $(IPGSQL)
#    - assumes that BLAS and LAPACK libraries are installed
#      lapack libraries are assumed to be in $(LAPACK)
#
	OS = linux
# SuSe:
	IPGSQL = -I/usr/include/pgsql
	LAPACK = -L/usr/lib64 -llapack -lblas
# Ubuntu:
#       IPGSQL = -I/usr/include/postgresql
#       LAPACK = -L/usr/lib -llapack -lblas
# RedHat:
#       IPGSQL = -I/usr/include/postgresql
#       LAPACK = -llapack -lrefblas
#       LIBS = -lm -lgfortran $(LAPACK) $(RSTTLIB)
endif

#
#
# Default build: Single-processor with PostgreSQL database interface
#
#
serial_db: EXEC = iscloc_rstt_db
serial_db: LIBS = -lm -lpq $(LAPACK) $(RSTTLIB)
serial_db: CFLAGS = $(DEBUG) -I$(ISCPATH) $(RSTTINC) $(IPGSQL) $(MACOS) -DWITH_DB=1 -DWITH_RSTT=1
serial_db: $(objs) $(iscloc_objs)
	$(CC) -o $(EXEC) $(objs) $(iscloc_objs) $(LIBS)
	rm $(objs) $(iscloc_objs)

#
#
# Single-processor without database interface
#
#
serial_nodb: EXEC = iscloc_rstt_nodb
serial_nodb: LIBS = -lm $(LAPACK) $(RSTTLIB)
serial_nodb: CFLAGS = $(DEBUG) -I$(ISCPATH) $(RSTTINC) $(MACOS) -DWITH_RSTT=1
serial_nodb: $(objs) $(iscloc_objs)
	$(CC) -o $(EXEC) $(objs) $(iscloc_objs) $(LIBS)
	rm $(objs) $(iscloc_objs)

#
# delete object files
#
clean:
	rm -f $(objs) $(iscloc_objs) *~

