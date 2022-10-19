#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "tensor.h"

// Transpose a matrix
// tensor m: matrix to be transposed
// returns: tensor, result of transposition
tensor matrix_transpose(tensor a)
{
    assert(a.n == 2);
    // TODO 1.0: return a transposed version of a (don't modify a)
    size_t s[2] = { a.size[1], a.size[0] };
    tensor t = tensor_make(a.n, s);
    // see: https://stackoverflow.com/questions/38627087/taking-the-transpose-of-a-matrix-in-c-with-1d-arrays
    size_t n = a.size[0];
    size_t m = a.size[1];
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t i1 = i * n + j;
            size_t i2 = j * m + i;
            t.data[i1] = a.data[i2];
        }
    }
    return t;
}

// Perform matrix multiplication a*b, return result
// tensor a,b: operands
// returns: new tensor that is the result
tensor matrix_multiply(const tensor a, const tensor b)
{
    assert(a.n == 2);
    assert(b.n == 2);
    assert(a.size[1] == b.size[0]);
    // TODO 1.1: matrix multiplication! just use 3 for loops
    // size = height matrix a * width matrix b
    size_t s[2] = { a.size[0], b.size[1] }; 
    tensor t = tensor_make(2, s);

    // see: https://stackoverflow.com/questions/47023651/multiplying-matrices-in-one-dimensional-arrays

    size_t a_n = a.size[0];
    size_t b_n = b.size[0];
    size_t a_m = a.size[1];
    size_t b_m = b.size[1];
    
    for (size_t i = 0; i < a_n; ++i) {
        for (size_t j = 0; j < b_m; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < b_n; ++k) {
                sum = sum + a.data[i * b_n + k] * b.data[k * b_m + j];
            }
            t.data[i * b_m + j] = sum;
        }
    }

    return t;
}

// Used for matrix inversion
tensor matrix_augment(tensor m)
{
    assert(m.n == 2);
    size_t rows = m.size[0];
    size_t cols = m.size[1];
    size_t i,j;
    tensor c = tensor_vmake(2, rows, cols*2);
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            c.data[i*cols*2 + j] = m.data[i*cols + j];
        }
    }
    for(j = 0; j < rows; ++j){
        c.data[j*cols*2 + j+cols] = 1;
    }
    return c;
}

// Invert matrix m
tensor matrix_invert(tensor m)
{
    size_t i, j, k;
    assert(m.n == 2);
    assert(m.size[0] == m.size[1]);

    tensor c = matrix_augment(m);
    tensor none = {0};
    float **cdata = calloc(c.size[0], sizeof(float *));
    for(i = 0; i < c.size[0]; ++i){
        cdata[i] = c.data + i*c.size[1];
    }

    for(k = 0; k < c.size[0]; ++k){
        float p = 0.;
        size_t index = -1;
        for(i = k; i < c.size[0]; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            tensor_free(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.size[1]; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.size[0]; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.size[0]-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    tensor inv = tensor_make(2, m.size);
    for(i = 0; i < m.size[0]; ++i){
        for(j = 0; j < m.size[1]; ++j){
            inv.data[i*m.size[1] + j] = cdata[i][j+m.size[1]];
        }
    }
    tensor_free(c);
    free(cdata);
    return inv;
}

tensor solve_system(tensor M, tensor b)
{
    tensor none = {0};
    tensor Mt = matrix_transpose(M);
    tensor MtM = matrix_multiply(Mt, M);
    tensor MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    tensor Mdag = matrix_multiply(MtMinv, Mt);
    tensor a = matrix_multiply(Mdag, b);
    tensor_free(Mt);
    tensor_free(MtM);
    tensor_free(MtMinv);
    tensor_free(Mdag);
    return a;
}
