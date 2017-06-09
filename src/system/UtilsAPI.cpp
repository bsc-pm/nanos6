#include <nanos6/utils.h>

#include <string.h>


void nanos6_bzero(void *buffer, size_t size)
{
	memset(buffer, 0, size);
}

