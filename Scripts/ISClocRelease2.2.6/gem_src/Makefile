#
# gemloc Makefile
#
# Istvan Bondar, ISC, September, 2011 - November 2013
#

#
# Path for iscloc source files
#
#ISCPATH = /export/home/istvan/ISCsrc/ISCloc/Multicore
ISCPATH = ../src

#
# object files
#
objs = read_db_gem.o \

iscloc_files = main.o \
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
               read_textfiles.o \
               svd_parallel.o \
               timefuncs.o \
               traveltimes_parallel.o \
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
#    - linux/unix: assumed that PostgreSL client is installed
#                  PostgreSQL include files are assumed to be in $(IPGSQL)
#   Grand Central Dispatch
#    - SLBM is not thread safe, cannot use Grand Central Dispatch 
#      for multicore processing

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
# Default build: Single-processor with PostgreSQL database interface
#
serial_db: EXEC = gemloc
serial_db: LIBS = -lm -lpq $(LAPACK)
serial_db: CFLAGS = $(DEBUG) -I$(ISCPATH) $(IPGSQL) $(MACOS) -DWITH_DB=1
serial_db: $(iscloc_objs) $(objs)
	$(CC) -o $(EXEC) $(iscloc_objs) $(objs) $(LIBS)
	rm $(objs) $(iscloc_objs)

#
# Single-processor without database interface
#
serial_nodb: EXEC = gemloc_nodb
serial_nodb: LIBS = -lm $(LAPACK)
serial_nodb: CFLAGS = $(DEBUG) -I$(ISCPATH) $(MACOS)
serial_nodb: $(iscloc_objs) $(objs)
	$(CC) -o $(EXEC) $(iscloc_objs) $(objs) $(LIBS)
	rm $(objs) $(iscloc_objs)

#
# GCD multicore processing with PostgreSQL database interface
#
ifeq ($(OS),darwin)
parallel_db: EXEC = gemloc_parallel_db
parallel_db: LIBS = -lm -lpq $(LAPACK)
parallel_db: CFLAGS = $(DEBUG) -I$(ISCPATH) $(MACOS) -DWITH_DB=1 -DWITH_GCD=1
parallel_db: $(iscloc_objs) $(objs)
	$(CC) -o $(EXEC) $(iscloc_objs) $(objs) $(LIBS)
	rm $(objs) $(iscloc_objs)
endif

#
# GCD multicore processing without database interface
#
ifeq ($(OS),darwin)
parallel_nodb: EXEC = gemloc_parallel_nodb
parallel_nodb: LIBS = -lm $(LAPACK)
parallel_nodb: CFLAGS = $(DEBUG) -I$(ISCPATH) $(MACOS) -DWITH_GCD=1
parallel_nodb: $(iscloc_objs) $(objs)
	$(CC) -o $(EXEC) $(iscloc_objs) $(objs) $(LIBS)
	rm $(objs) $(iscloc_objs)
endif


clean:
	rm -f $(objs) $(iscloc_objs) *~

