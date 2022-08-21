#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mmio.h"

#define BlockDim 1024
#define ITER 3


__global__ void
spmv_ell_kernel ( const int num_rows , const int num_cols , const int num_cols_per_row , const int * indices , const float * data , const float * x , float * y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if( row < num_rows ){
        float dot = 0;
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = indices [ num_rows * n + row ];
            float val = data [ num_rows * n + row ];
            if( val != 0)
                dot += val * x [ col ];
        }
        y[ row ] += dot ;
    }
}