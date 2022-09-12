/**
 *		Tempesta DB
 *
 * Generic storage layer.
 *
 * Copyright (C) 2014 NatSys Lab. (info@natsys-lab.com).
 * Copyright (C) 2015-2018 Tempesta Technologies, INC.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */
#ifndef __TDB_H__
#define __TDB_H__

#ifdef __KERNEL__
#include <linux/fs.h>
#include <linux/slab.h>

#include "tdb_if.h"
#else
#include "kernel_mocks.h"
#endif

#include "tdb_internal.h"

#ifdef __KERNEL__
/**
 * Database handle descriptor.
 *
 * @filp	- mmap()'ed file;
 * @node	- NUMA node ID;
 * @count	- reference counter;
 * @ga_lock	- Lock for atomic execution of lookup and create a record TDB;
 * @tbl_name	- table name;
 * @path	- path to the table;
 */
typedef struct {
	TdbHdr		*hdr;
	struct file	*filp;
	int		node;
	atomic_t	count;
	spinlock_t	ga_lock; /* TODO: remove with movement to the new HTrie. */
	char		tbl_name[TDB_TBLNAME_LEN + 1];
	char		path[TDB_PATH_LEN];
} TDB;
#endif

/**
 * Fixed-size (and typically small) records.
 *
 * TODO it seems we need reference counting for concurrent removal or use RCU
 *
 * @data - might be larger than 8 bytes, must be the last item.
 * @off - byte offset of the record with preserved address.
 */
typedef struct {
	unsigned long		key; /* must be the first */
	union {
		char		data[8];
		unsigned long	off;
	};
} __attribute__((packed)) TdbFRec;

/**
 * Variable-size (typically large) record.
 * This type of records is always referenced by metadata blocks.
 *
 * @chunk_next - index of next data chunk
 * @len - data length of current chunk
 */
typedef struct {
	unsigned int	chunk_next;
	unsigned int	len;
} __attribute__((packed)) TdbVRec;

/* Common interface for database records of all kinds. */
typedef TdbFRec TdbRec;

/**
 * Iterator for TDB full key collision chains.
 */
typedef struct {
	TdbRec	*rec;
	void	*bckt;
} TdbIter;

#define TDB_ITER_BAD(i)		(!(i).rec)

/**
 * Hooks for tdb_rec_get_alloc() function.
 * @eq_rec		- record match function, used in collision chain;
 * @precreate_rec	- called before a new record will be created int tdb,
 *			record creation will be aborted in non zero return code;
 * @init_rec		- init record before use;
 * @ctx			- arbitrary pointer to pass arguments into callbacks;
 * @len			- requested and resulting record size;
 * @is_new		- true if entry wasn't found in tdb and a new one was
 *			  created;
 *
 * All function pointers get @ctx as argument. If @init_rec fail the
 * record is already created and placed into tdb. Tdb user is responsible to
 * deal with invalid records.
 */
typedef struct {
	bool		(*eq_rec)(TdbRec *rec, void *ctx);
	int		(*precreate_rec)(void *ctx);
	void		(*init_rec)(TdbRec *rec, void *ctx);
	void		*ctx;
	size_t		len;
	bool		is_new;
} TdbGetAllocCtx;

/**
 * We use very small index nodes size of only one cache line.
 * So overall memory footprint of the index is minimal by a cost of more LLC
 * or main memory transfers. However, smaller memory usage means better TLB
 * utilization on huge worksets.
 */
#define TDB_HTRIE_NODE_SZ	L1_CACHE_BYTES
/*
 * There is no sense to allocate a new resolving node for each new small
 * (less than cache line size) data record. So we place small records in
 * 2 cache lines in sequential order and burst the node only when there
 * is no room.
 */
#define TDB_HTRIE_MINDREC	(L1_CACHE_BYTES * 2)

/* Get index and data block indexes by byte offset and vise versa. */
#define TDB_O2DI(o)		((o) / TDB_HTRIE_MINDREC)
#define TDB_O2II(o)		((o) / TDB_HTRIE_NODE_SZ)
#define TDB_DI2O(i)		((i) * TDB_HTRIE_MINDREC)
#define TDB_II2O(i)		((i) * TDB_HTRIE_NODE_SZ)

#define TDB_BANNER		"[tdb] "

/*
 * Tempesta DB is too internal piece of code, so print its messages on
 * higher debugging levels.
 */
#if defined(DEBUG) && (DEBUG >= 2)
#define TDB_DBG(...)		pr_debug(TDB_BANNER "  " __VA_ARGS__)
#else
#define TDB_DBG(...)
#endif
#define TDB_LOG(...)		pr_info(TDB_BANNER __VA_ARGS__)
#define TDB_WARN(...)		pr_warn(TDB_BANNER "Warning: " __VA_ARGS__)
#define TDB_ERR(...)		pr_err(TDB_BANNER "ERROR: " __VA_ARGS__)

#ifdef __KERNEL__
/*
 * Storage routines.
 *
 * BEWARE(!) the routines use SIMD instructions, so protect them with
 * kernel_fpu_begin()/kernel_fpu_end() or call from softirq context only.
 */
TdbRec *tdb_entry_alloc(TDB *db, unsigned long key, size_t *len);
TdbRec *tdb_entry_create(TDB *db, unsigned long key, void *data, size_t *len);
TdbVRec *tdb_entry_add(TDB *db, TdbVRec *r, size_t size);
void *tdb_entry_get_room(TDB *db, TdbVRec **r, char *curr_ptr, size_t tail_len,
			 size_t tot_size);
TdbIter tdb_rec_get(TDB *db, unsigned long key);
void tdb_rec_next(TDB *db, TdbIter *iter);

/*
 * Release a read-lock on the record's bucket.
 */
void tdb_rec_put(void *rec);

/*
 * Acquire a read-lock on the record's bucket.
 *
 * Record iteration functions take a read-lock that is released when iteration
 * finishes, or when moving between buckets. Use this function, if you need to
 * keep the record locked.
 *
 * TODO #515: the locking interface should be apperently reworked with the new
 * lock free HTrie buckets.
 */
void tdb_rec_keep(void *rec);

int tdb_info(char *buf, size_t len);
TdbRec * tdb_rec_get_alloc(TDB *db, unsigned long key, TdbGetAllocCtx *ctx);
int tdb_entry_walk(TDB *db, int (*fn)(void *));
void tdb_rec_get_lock(void *rec);

/* Open/close database handler. */
TDB *tdb_open(const char *path, size_t fsize, unsigned int rec_size, int node);
void tdb_close(TDB *db);

static inline TDB *
tdb_get(TDB *db)
{
	atomic_inc(&db->count);
	return db;
}

static inline void
tdb_put(TDB *db)
{
	if (atomic_dec_and_test(&db->count))
		kfree(db);
}

static inline TdbVRec *
tdb_next_rec_chunk(TDB *db, TdbVRec *r)
{
	if (!r->chunk_next)
		return NULL;
	return TDB_PTR(db->hdr, TDB_DI2O(r->chunk_next));
}
#endif

#endif /* __TDB_H__ */