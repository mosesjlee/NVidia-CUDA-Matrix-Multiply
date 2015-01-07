#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

//#define WIDTH 32 // Matrix width
#ifndef GLOBAL
#define GLOBAL 0
#endif
#ifndef ROW_SIZE
#define ROW_SIZE 32 // divides matrix width
#endif
#ifndef COL_SIZE
#define COL_SIZE 32 // divides matrix width
#endif
#ifndef K_SIZE
#define K_SIZE 32 // divides matrix width
#endif
#define THREAD_BLOCK 16 // divdes width, ROW_SIZE, K_SIZE evenly
#endif // _MATRIXMUL_H_
