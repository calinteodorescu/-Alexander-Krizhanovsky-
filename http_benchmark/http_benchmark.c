/**
 * Benchmark for traditional switch-driven HTTP state machine (the code
 * is mainly borrowed from Nginx) with HTTP Hybrid State Machine.
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
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

#include "http.h"

#ifdef UNALIGNED
#define STR(s)	{" " s, sizeof(s) - 1}
#define OFF	1
#define	N	(20000 * 1000)
#else
#define STR(s)	{s, sizeof(s) - 1}
#define OFF	0
#define	N	(500 * 1000)
#endif

static struct {
	const char	*str;
	size_t 		len;
} headers[10] = {
	STR("Host: github.com\r\n"),
	STR("Connection: keep-alive\r\n"),
	STR("Cache-Control: max-age=0\r\n"),
	STR("Upgrade-Insecure-Requests: 1\n"),
	STR("Accept-Encoding: gzip,deflate,sdch\r\n"),
	STR("Accept-Language: zh-CN,zh;q=0.8,en;q=0.6\r\n"),
	STR("Accept-Charset: gb18030,utf-8;q=0.7,*;q=0.3\r\n"),
	STR("If-None-Match: 7f9c6a2baf61233cedd62ffa906b604f\r\n"),
	STR("User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11\r\n"),
	STR("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n")
},

requests[10] = {
	STR("GET / HTTP/1.1\r\n"),
	STR("GET /index.html HTTP/1.0\r\n"),
	STR("COPY /foo/bar/zoo HTTP/1.1\r\n"),
	STR("GET ftp://mail.ru/index.html HTTP/1.1\r\n"),
	STR("POST /script1?a=44,fd=6 HTTP/1.1\r\n"),
	STR("GET /joyent/http-parser HTTP/1.1\r\n"),
	STR("PUT   http://mail.ru/index.html HTTP/1.1\r\n"),
	STR("POST /api/2/thread/404435440?1340553000964 HTTP/1.1\r\n"),
	STR("PROPFIND /foo/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q HTTP/1.1\r\n"),
	STR("GET http://pipelined-host-C.co.uk/s/o/m/e/p/a/g/e.abc/hjkhasdfdaf$#ffse4wds HTTP/1.1\n"),
};

static inline unsigned long
tv_to_ms(const struct timeval *tv)
{
	return ((unsigned long)tv->tv_sec * 1000000 + tv->tv_usec) / 1000;
}

static ngx_http_request_t r;

#define test(data, fn)							\
do {									\
	struct timeval tv0, tv1;					\
									\
	gettimeofday(&tv0, NULL);					\
									\
	for (int i = 0; i < N; ++i)					\
		for (int j = 0; j < sizeof(data)/sizeof(data[0]); ++j) { \
			r.state = 0;					\
			r.__state = NULL;				\
			fn(&r, (unsigned char *)data[j].str + OFF,	\
			   data[j].len);				\
		}							\
									\
	gettimeofday(&tv1, NULL);					\
									\
	printf("\t" #fn ":\t%lums\n", tv_to_ms(&tv1) - tv_to_ms(&tv0));	\
} while (0)

int
main()
{
	printf("Nginx HTTP parser:\n");
	test(requests, ngx_request_line);
	test(headers, ngx_header_line);
	test(headers, ngx_big_header_line);

	/* Disable PoC parsers. */
#if 0
	printf("\nHTTP Hybrid State Machine:\n");
	test(headers, hsm_header_line);

	printf("\nTable-driven Automaton (DPI)\n");
	test(headers, tbl_header_line);
	test(headers, tbl_big_header_line);
#endif
	printf("\nGoto-driven Automaton:\n");
	test(requests, goto_request_line);
	test(headers, goto_header_line);
	test(headers, goto_big_header_line);

	printf("\n[req: http/%d.%d: %d %p %p %p %p %p %p]\n\n",
		r.http_major, r.http_minor,
		r.state, r.header_name_start,
		r.header_start, r.request_start, r.schema_start,
		r.uri_start, r.host_start);

	return 0;
}
