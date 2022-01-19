#ifndef STRING_H
#define STRING_H

#include "PR/ultratypes.h"

size_t strlen(const char *str);
char *strchr(const char *str, s32 ch);
/*


void *memcpy(int *dst, const int *src, size_t size) {
    u8 *_dst = dst;
    for(int i=0;i<(size>>2);++i){
			dst[i]=src[i];
	}
	
    return dst;
}

void * memset ( int * ptr, int value, size_t num ){
	int v2=value<<24|value<<16|value<<8|value;
	for(unsigned short i=0;i<(num>>2);++i){
			ptr[i]=value;
	}
	return ptr;
}
*/
#endif
