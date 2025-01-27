#		Benchmark for lock-free data structures
#
# Copyright (C) 2016-2022 Alexander Krizhanovsky (ak@tempesta-tech.com).
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59
# Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
CFLAGS	= -march=native -mtune=native -ltbbmalloc_proxy \
	  -Wno-address-of-packed-member
ifdef DEBUG
	CFLAGS += -O0 -ggdb3
	CFLAGS += -DDEBUG=3 # Tempesta DB debugging is printed on the level >=2
else
	CFLAGS += -O3 -DNDEBUG
endif
ifdef BIG
	CFLAGS += -DBIG_MACHINE=1
endif
CXXFLAGS = $(CFLAGS) -std=c++20 -Wno-register

all: lfds_bench test

lfds_bench: benchmark.o htrie.o mapfile.o alloc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread -ltbb

test: test.o htrie.o alloc.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread

benchmark.o : benchmark.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^ -lpthread -ltbb

test.o : test.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $^ -lpthread

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean : FORCE
	rm -f *.o* *~ lfds_bench test

FORCE :

