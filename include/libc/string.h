#ifndef STRING_H
#define STRING_H

#include "PR/ultratypes.h"

void *memcpy(int *dst, const int *src, size_t size){
	size/=4;
	for(unsigned int i=0;i<size;++i){
			dst[i]=src[i];
	}
	
}
size_t strlen(const char *str);
char *strchr(const char *str, s32 ch);

#endif
