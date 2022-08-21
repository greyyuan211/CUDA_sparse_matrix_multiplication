#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mmio.h"

#define BlockDim 1024
#define ITER 3


__global__ void spmv_dia_kernel ( const int num_rows ,const int num_cols , const int num_diags , const int * offsets , 
                                    const float * data , const float * x , float * y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if( row < num_rows ){
        float dot = 0;
        for ( int n = 0; n < num_diags ; n ++){
            int col = row + offsets [n ];
            float val = data [ num_rows * n + row ];
            if( col >= 0 && col < num_cols )
                dot += val * x [ col ];
        }
        y[ row ] += dot ;
    }
}


