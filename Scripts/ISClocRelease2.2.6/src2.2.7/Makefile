#
# iscloc Makefile
#
# Istvan Bondar, ISC, August 2012 - November 2013
#

#
# object files
#
objs    =       main.o \
                calc_ellipticity_corr.o \
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
                phaseids_parallel.o \
                print_event.o \
                read_db.o \
                read_textfiles.o \
                svd_parallel.o \
                timefuncs.o \
                traveltimes_parallel.o \
                utils.o \
                write_db.o

#
# default compiler
#    (on Mac OS gcc is a symbolic link to llvm-gcc)
#
CC = gcc
DEBUG = -Wall -g -O0

#
# Determine OS and set OS-specific compiler variables
#
OS := $(shell uname -s)
ifeq ($(OS),Darwin)
#
# Mac OS X 10.6 and later:
#    - Grand Central Dispatch and PostgreSQL client are part of the OS
#    - LAPACK is part of the Accelerate framework
#
#    - multicore processing:
#      use
#          make parallel_db or
#          make parallel_nodb
#      enables Grand Central Dispatch for parallel processing
#
	OS = darwin
	LAPACK = -framework Accelerate
	MACOS = -DMACOSX=1
# Mac OS X 10.9 and later:
#    - PostgreSQL client was removed from the OS
	IPGSQL = -I../postgresql_include
else
#
# Linux (tested on OpenSUSE 10.3 and Ubuntu):
#    - assumes that PostgreSL client is installed
#      PostgreSQL include files are assumed to be in $(IPGSQL)
#    - assumes that BLAS and LAPACK libraries are installed
#      lapack libraries are assumed to be in $(LAPACK)
#
#    - multicore processing (so far tested only on Ubuntu):
#      use
#          make parallel_db or
#          make parallel_nodb
#      enables Grand Central Dispatch for parallel processing
#      uses clang instead of gcc
#      assumes that clang and llvm are installed
#          see llvm.org
#      assumes that libBlocksRuntime, libkqueue, libpthread_workqueue and
#          libdispatch are installed
#          see http://mark.heily.com/project/libblocksruntime
#
	OS = linux
# SuSe:
	#IPGSQL = -I/usr/include/pgsql
	IPGSQL = -I$(CONDA_PREFIX)/include  #**changed to be conda installed and point to conda env
	#LAPACK = -L/usr/lib64 -llapack -lblas
	LAPACK = -L$(CONDA_PREFIX)/lib -llapack -lblas -lgfortran -Wl,-rpath,$(CONDA_PREFIX)/lib #**changed to be conda installed and point to conda env
# Ubuntu:
#	IPGSQL = -I/usr/include/postgresql
#	LAPACK = -L/usr/lib -llapack -lblas
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
serial_db: EXEC = iscloc
serial_db: LIBS = -lm -lpq $(LAPACK)
serial_db: CFLAGS = $(DEBUG) $(IPGSQL) $(MACOS) -DWITH_DB=1
serial_db: $(objs)
	$(CC) -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)
	
#
#
# Single-processor without database interface
#
#
serial_nodb: EXEC = iscloc_nodb
serial_nodb: LIBS = -lm $(LAPACK)
serial_nodb: CFLAGS = $(DEBUG) $(MACOS)
serial_nodb: $(objs)
	$(CC) -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)

#
#
# GCD multicore processing with PostgreSQL database interface
#
#
ifeq ($(OS),darwin)
#
# Mac OS 10.6 and later
#
parallel_db: EXEC = iscloc_parallel_db
parallel_db: LIBS = -lm -lpq $(LAPACK)
parallel_db: CFLAGS = $(DEBUG) $(IPGSQL) $(MACOS) -DWITH_DB=1 -DWITH_GCD=1
parallel_db: $(objs)
	$(CC) -v -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)
else
#
# linux
#
parallel_db: CC = clang
parallel_db: EXEC = iscloc_parallel_db
parallel_db: LIBS = -lm -lpq $(LAPACK) -ldispatch -lBlocksRuntime
parallel_db: CFLAGS = -fblocks $(DEBUG) $(IPGSQL) -DWITH_DB=1 -DWITH_GCD=1
parallel_db: $(objs)
	$(CC) -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)
endif

#
#
# GCD multicore processing without database interface
#
#
ifeq ($(OS),darwin)
#
# Mac OS 10.6 and later
#
parallel_nodb: EXEC = iscloc_parallel_nodb
parallel_nodb: LIBS = -lm $(LAPACK)
parallel_nodb: CFLAGS = $(DEBUG) $(MACOS) -DWITH_GCD=1
parallel_nodb: $(objs)
	$(CC) -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)
else
#
# linux
#
parallel_nodb: CC = clang
parallel_nodb: EXEC = iscloc_parallel_nodb
parallel_nodb: LIBS = -lm $(LAPACK) -ldispatch -lBlocksRuntime
parallel_nodb: CFLAGS = -fblocks $(DEBUG) -DWITH_GCD=1
parallel_nodb: $(objs)
	$(CC) -o $(EXEC) $(objs) $(LIBS)
	rm $(objs)
endif

#
# delete object files after successful compilation
#
clean:
	rm -f $(objs) *~

