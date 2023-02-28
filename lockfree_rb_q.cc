/**
 * Implementation of Naive and Lock-free ring-buffer queues and
 * performance and verification tests.
 *
 * Build with (g++ version must be >= 4.5.0):
 * $ g++ -Wall -std=c++0x -Wl,--no-as-needed -O2 -D DCACHE1_LINESIZE=`getconf LEVEL1_DCACHE_LINESIZE` lockfree_rb_q.cc -lpthread
 *
 * I verified the program with g++ 4.5.3, 4.6.1, 4.6.3 and 4.8.1.
 *
 * Use -std=c++11 instead of -std=c++0x for g++ 4.8.
 *
 * Copyright (C) 2012-2013 Alexander Krizhanovsky (ak@tempesta-tech.com).
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
#ifndef __x86_64__
#warning "The program is developed for x86-64 architecture only."
#endif
#if !defined(DCACHE1_LINESIZE) || !DCACHE1_LINESIZE
#ifdef DCACHE1_LINESIZE
#undef DCACHE1_LINESIZE
#endif
#define DCACHE1_LINESIZE 64
#endif
#define ____cacheline_aligned    __attribute__((aligned(DCACHE1_LINESIZE)))

#include <sys/time.h>
#include <limits.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <immintrin.h>

#include <atomic>
#include <cassert>
#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>

#define QUEUE_SIZE    (32 * 1024)

/*
 * ------------------------------------------------------------------------
 * Naive serialized ring buffer queue
 * ------------------------------------------------------------------------
 */
template<class T, unsigned long Q_SIZE = QUEUE_SIZE>
class NaiveQueue {
private:
    static const unsigned long Q_MASK = Q_SIZE - 1;

public:
    NaiveQueue()
        : head_(0), tail_(0)
    {
        ptr_array_ = (T **)::memalign(getpagesize(),
                Q_SIZE * sizeof(void *));
        assert(ptr_array_);
    }

    void
    push(T *x)
    {
        std::unique_lock<std::mutex> lock(mtx_);

        cond_overflow_.wait(lock, [this]() {
                    return tail_ + Q_SIZE > head_;
                });

        ptr_array_[head_++ & Q_MASK] = x;

        cond_empty_.notify_one();
    }

    T *
    pop()
    {
        std::unique_lock<std::mutex> lock(mtx_);

        cond_empty_.wait(lock, [this]() {
                    return tail_ < head_;
                });

        T *x = ptr_array_[tail_++ & Q_MASK];

        cond_overflow_.notify_one();

        return x;
    }

private:
    unsigned long        head_, tail_;
    std::condition_variable    cond_empty_;
    std::condition_variable    cond_overflow_;
    std::mutex        mtx_;
    T            **ptr_array_;
};

/*
 * ------------------------------------------------------------------------
 * Boost lock-free fixed size multi-producer multi-consumer queue
 * ------------------------------------------------------------------------
 */
#include <boost/lockfree/queue.hpp>

template<class T, unsigned long Q_SIZE = QUEUE_SIZE>
class BoostQueue {
public:
    void
    push(T *x)
    {
        while (!q_.push(x))
            asm volatile("rep; nop" ::: "memory");
    }

    T *
    pop()
    {
        T *x;
        while (!q_.pop(x))
            asm volatile("rep; nop" ::: "memory");
        return x;
    }

private:
    boost::lockfree::queue<T *, boost::lockfree::capacity<Q_SIZE>> q_;
};


/*
 * ------------------------------------------------------------------------
 * Lock-free N-producers M-consumers ring-buffer queue.
 * ABA problem safe.
 *
 * This implementation is bit complicated, so possibly it has sense to use
 * classic list-based queues. See:
 * 1. D.Fober, Y.Orlarey, S.Letz, "Lock-Free Techniques for Concurrent
 *    Access to Shared Objects"
 * 2. M.M.Michael, M.L.Scott, "Simple, Fast and Practical Non-Blocking and
 *    Blocking Concurrent Queue Algorithms"
 * 3. E.Ladan-Mozes, N.Shavit, "An Optimistic Approach to Lock-Free FIFO Queues"
 *
 * See also implementation of N-producers M-consumers FIFO and
 * 1-producer 1-consumer ring-buffer from Tim Blechmann:
 *    http://tim.klingt.org/boost_lockfree/
 *    git://tim.klingt.org/boost_lockfree.git
 * 
 * See Intel 64 and IA-32 Architectures Software Developer's Manual,
 * Volume 3, Chapter 8.2 Memory Ordering for x86 memory ordering guarantees.
 * ------------------------------------------------------------------------
 */
static size_t __thread __thr_id;

/**
 * @return continuous thread IDs starting from 0 as opposed to pthread_self().
 */
inline size_t
thr_id()
{
    return __thr_id;
}

inline void
set_thr_id(size_t id)
{
    __thr_id = id;
}

template< typename           T,
          decltype( thr_id ) ThrId = thr_id,
          unsigned long      Q_SIZE = QUEUE_SIZE
        >
class LockFreeQueue 
{
private:
    static const unsigned long Q_MASK = Q_SIZE - 1;

    struct ThrPos 
    {
        unsigned long m_reservedOffsetToWriteTo  = 0;
        unsigned long m_reservedOffsetToReadFrom = 0;
    };

public:
    LockFreeQueue( size_t producers, 
                   size_t consumers
                 )
    :   m_producers( producers ),
        m_consumers( consumers )
    {
        auto n = std::max(m_consumers, m_producers);
        thr_p_ = (ThrPos *)::memalign(getpagesize(), sizeof(ThrPos) * n);
        assert(thr_p_);
        // Set per thread tail and head to ULONG_MAX.
        ::memset((void *)thr_p_, 0xFF, sizeof(ThrPos) * n);

        m_storage = ( T** )::memalign( getpagesize( ),
                                       Q_SIZE * sizeof( void* )
                                     );
        assert( m_storage );
    }

    ~LockFreeQueue()
    {
        ::free( m_storage );
        ::free(thr_p_);
    }

    ThrPos&
    thr_pos() const
    {
        assert(ThrId() < std::max(m_consumers, m_producers));
        return thr_p_[ThrId()];
    }

    auto GetEarliestOffsetReservedToReadFrom( void )
    {
        auto min = m_nextAvailableOffsetToReadFrom;
        for( size_t i = 0
             ;
             i < m_consumers
             ;
             ++i
           ) 
        {
            auto tmp_t = thr_p_[ i ].m_reservedOffsetToReadFrom;

            // Force compiler to use tmp_t exactly once.
            asm volatile("" ::: "memory");

            if ( tmp_t < min )
                min = tmp_t;
        }

        return min;
    }

    void LightlyWaitToReallyHaveRoom( void )
    {
        /*
         * We do not know when a consumer uses the pop()'ed pointer,
         * so we can not overwrite it and have to wait the lowest tail.
         */

        ThrPos& tp = thr_pos( );
        for( bool overflown = false
             ;
             overflown = __builtin_expect( tp.m_reservedOffsetToWriteTo >= m_theEarliestReservedSlotToReadFrom + Q_SIZE, 0 )
             ;
           )
        {
            // Update the m_theEarliestReservedSlotToReadFrom
            m_theEarliestReservedSlotToReadFrom = GetEarliestOffsetReservedToReadFrom( );

            bool hasRoom = ( tp.m_reservedOffsetToWriteTo < m_theEarliestReservedSlotToReadFrom + Q_SIZE );
            if ( hasRoom )
                break;

            _mm_pause();
        }
    }

    void ReserveWriteToOffset( void )
    {
        ThrPos& tp = thr_pos( );

        // announce non-atomic the floor of Owned Offset to Write To
        tp.m_reservedOffsetToWriteTo = m_nextAvailableOffsetToWriteTo;
        // reserve the offset, maybe higher than the older value because meantime other threads pushed Data
        tp.m_reservedOffsetToWriteTo = __sync_fetch_and_add( & m_nextAvailableOffsetToWriteTo, 1 );
    }

    void push( T* val )
    {
        /*
         * Request next place to push.
         *
         * Second assignment is atomic only for head shift, so there is
         * a time window in which thr_p_[tid].head = ULONG_MAX, and
         * head could be shifted significantly by other threads,
         * so pop() will set m_theEarliestReservedSlotToWriteTo to m_nextAvailableOffsetToWriteTo.
         * After that thr_p_[tid].head is set to old head value
         * (which is stored in local CPU register) and written by @ptr.
         *
         * First assignment guaranties that pop() sees values for
         * head and thr_p_[tid].head not greater that they will be
         * after the second assignment with head shift.
         *
         * Loads and stores are not reordered with locked instructions,
         * so we don't need a memory barrier here.
         */

        ThrPos& tp = thr_pos( );
        
        ReserveWriteToOffset       ( );
        LightlyWaitToReallyHaveRoom( );

        m_storage[ tp.m_reservedOffsetToWriteTo & Q_MASK ] = val;

        // Allow consumers to eat the item.
        tp.m_reservedOffsetToWriteTo = ULONG_MAX;
    }

    auto GetEarliestOffsetReservedToWriteTo( void )
    {
        ThrPos& tp = thr_pos( );

        auto min = m_nextAvailableOffsetToWriteTo;
        for( size_t i = 0
             ;
             i < m_producers
             ;
             ++i
           )
        {
            auto tmp_h = thr_p_[ i ].m_reservedOffsetToWriteTo;

            // Force compiler to use tmp_h exactly once.
            asm volatile("" ::: "memory");

            if ( tmp_h < min )
                min = tmp_h;
        }

        return min;
    }

    void LightlyWaitToReallyHaveStuffToRead( void )
    {
        /*
         * tid'th place in m_storage is reserved by the thread -
         * this place shall never be rewritten by push() and
         * m_theEarliestReservedSlotToWriteTo at push() is a guarantee.
         * m_theEarliestReservedSlotToWriteTo guaranties that no consumer eats the item
         * before producer reserved the position writes to it.
         */

        ThrPos& tp = thr_pos( );

        for( bool nothingToRead = false;
             ;
             nothingToRead = __builtin_expect( tp.m_reservedOffsetToReadFrom >= m_theEarliestReservedSlotToWriteTo, 0 )
             ;
           )
        {
            // Update the m_theEarliestReservedSlotToWriteTo.
            m_theEarliestReservedSlotToWriteTo = GetEarliestOffsetReservedToWriteTo( );

            bool hasStuffToRead = ( tp.m_reservedOffsetToReadFrom < m_theEarliestReservedSlotToWriteTo );
            if ( hasStuffToRead )
                break;

            _mm_pause();
        }
    }

    void ReserveReadFromOffset( void )
    {
        ThrPos& tp = thr_pos( );

        // announce non-atomic the floor of Owned Offset to Read From
        tp.m_reservedOffsetToReadFrom = m_nextAvailableOffsetToReadFrom;
        // reserve the offset, maybe higher than the older value because meantime other threads popped Data
        tp.m_reservedOffsetToReadFrom = __sync_fetch_and_add( & m_nextAvailableOffsetToReadFrom, 1);
    }

    T* pop( void )
    {
        assert(ThrId() < std::max(m_consumers, m_producers));

        ThrPos& tp = thr_pos( );
        /*
         * Request next place from which to pop.
         * See comments for push().
         *
         * Loads and stores are not reordered with locked instructions,
         * so we don't need a memory barrier here.
         */
        
        ReserveReadFromOffset             ( );
        LightlyWaitToReallyHaveStuffToRead( );

        // read from the reserved offset
        T* val = m_storage[ tp.m_reservedOffsetToReadFrom & Q_MASK ];

        // Allow producers rewrite the slot.
        tp.m_reservedOffsetToReadFrom = ULONG_MAX;

        return val;
    }

private:
    /*
     * The most hot members are cacheline aligned to avoid
     * False Sharing.
     */

    const size_t     m_producers;
    const size_t     m_consumers;
    // currently free position (next to insert)
    unsigned long    m_nextAvailableOffsetToWriteTo      ____cacheline_aligned = 0;
    // current m_nextAvailableOffsetToReadFrom, next to pop
    unsigned long    m_nextAvailableOffsetToReadFrom     ____cacheline_aligned = 0;
    // last not-processed producer's pointer
    unsigned long    m_theEarliestReservedSlotToWriteTo  ____cacheline_aligned = 0;
    // last not-processed consumer's pointer
    unsigned long    m_theEarliestReservedSlotToReadFrom ____cacheline_aligned = 0;
    ThrPos*          thr_p_;
    T**              m_storage;
};


/*
 * ------------------------------------------------------------------------
 *    Tests for naive and lock-free queues
 * ------------------------------------------------------------------------
 */
static const auto N         = QUEUE_SIZE * 1024;
static const auto CONSUMERS = 2;
static const auto PRODUCERS = 2;

typedef unsigned char    q_type;

static const q_type X_EMPTY = 0; // the address skipped by producers
static const q_type X_MISSED = 255; // the address skipped by consumers
q_type x[N * PRODUCERS];
std::atomic<int> n(0);

template<class Q>
struct Worker {
    Worker(Q *q, size_t id = 0)
        : q_(q),
        thr_id_(id)
    {}

    Q *q_;
    size_t thr_id_;
};

template<class Q>
struct Producer : public Worker<Q> {
    Producer(Q *q, size_t id)
        : Worker<Q>(q, id)
    {}

    void operator()()
    {
        set_thr_id(Worker<Q>::thr_id_);

        for (auto i = thr_id(); i < N * PRODUCERS; i += PRODUCERS) {
            x[i] = X_MISSED;
            Worker<Q>::q_->push(x + i);
        }
    }
};

template<class Q>
struct Consumer : public Worker<Q> {
    Consumer(Q *q, size_t id)
        : Worker<Q>(q, id)
    {}

    void operator()()
    {
        set_thr_id(Worker<Q>::thr_id_);

        while (n.fetch_add(1) < N * PRODUCERS) {
            q_type *v = Worker<Q>::q_->pop();
            assert(v);
            assert(*v == X_MISSED);
            *v = (q_type)(thr_id() + 1); // don't write zero
        }
    }
};

static inline unsigned long
tv_to_ms(const struct timeval &tv)
{
    return ((unsigned long)tv.tv_sec * 1'000'000 + tv.tv_usec) / 1'000;
}

template<class Q>
void
run_test(Q &&q)
{
    std::thread thr[PRODUCERS + CONSUMERS];

    n.store(0);
    ::memset(x, X_EMPTY, N * sizeof(q_type) * PRODUCERS);

    struct timeval tv0, tv1;
    gettimeofday(&tv0, NULL);

    // Run producers.
    for (auto i = 0; i < PRODUCERS; ++i)
        thr[i] = std::thread(Producer<Q>(&q, i));

    ::usleep(10 * 1'000); // sleep to wait the queue is full

    /*
     * Run consumers.
     * Create consumers with the same thread IDs as producers.
     * The IDs are used for queue head and tail indexing only,
     * so we  care only about different IDs for threads of the same type.
     */
    for (auto i = 0; i < CONSUMERS; ++i)
        thr[PRODUCERS + i] = std::thread(Consumer<Q>(&q, i));

    // Wait for all threads completion.
    for (auto i = 0; i < PRODUCERS + CONSUMERS; ++i)
        thr[i].join();

    gettimeofday(&tv1, NULL);
    std::cout << (tv_to_ms(tv1) - tv_to_ms(tv0)) << "ms" << std::endl;

    // Check data.
    auto res = 0;
    std::cout << "check X data..." << std::endl;
    for (auto i = 0; i < N * PRODUCERS; ++i) {
        if (x[i] == X_EMPTY) {
            std::cout << "empty " << i << std::endl;
            res = 1;
            break;
        } else if (x[i] == X_MISSED) {
            std::cout << "missed " << i << std::endl;
            res = 2;
            break;
        }
    }
    std::cout << (res ? "FAILED" : "Passed") << std::endl;
}

int
main()
{
    LockFreeQueue<q_type> lf_q(PRODUCERS, CONSUMERS);
    run_test<LockFreeQueue<q_type>>(std::move(lf_q));

    NaiveQueue<q_type> n_q;
    run_test<NaiveQueue<q_type>>(std::move(n_q));

    BoostQueue<q_type> b_q;
    run_test<BoostQueue<q_type>>(std::move(b_q));

    return 0;
}
