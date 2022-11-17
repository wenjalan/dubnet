#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "dubnet.h"
#include "test.h"
#include "tensor.h"
#include "matrix.h"

int tests_total = 0;
int tests_fail = 0;

double gflops(double ops, double time)
{
    return ops / time / pow(10., 9);
}

double currtime()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int within_eps(float a, float b)
{
    return fabs(a - b) < EPS;
}

int same_tensor(tensor a, tensor b)
{
    if (a.n != b.n)
    {
        fprintf(stderr, "Different dimensionality: %ld vs %ld\n", a.n, b.n);
        return 0;
    }
    size_t i;
    for (i = 0; i < a.n; ++i)
    {
        if (a.size[i] != b.size[i])
        {
            fprintf(stderr, "Dimension %ld, different size: %ld vs %ld\n", i, a.size[i], b.size[i]);
            return 0;
        }
    }
    size_t len = tensor_len(a);
    for (i = 0; i < len; ++i)
    {
        if (!within_eps(a.data[i], b.data[i]))
        {
            fprintf(stderr, "Different data at index %ld: %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

void test_tensor_make_get()
{
    tensor a = tensor_vmake(1, 1);
    tensor b = tensor_vmake(3, 3, 1080, 1920);
    tensor r = tensor_random(1.0f, 3, b.size);
    tensor srand = tensor_vrandom(1, 0);
    tensor smake = tensor_vmake(0);

    TEST(srand.n == 0);
    TEST(srand.size == 0);
    TEST(tensor_len(srand) == 1);

    TEST(smake.n == 0);
    TEST(smake.size == 0);
    TEST(tensor_len(smake) == 1);

    TEST(a.n == 1);
    TEST(a.size[0] = 1);

    TEST(b.n == 3);
    TEST(b.size[0] == 3);
    TEST(b.size[1] == 1080);
    TEST(b.size[2] == 1920);

    TEST(r.n == 3);
    TEST(r.size[0] == 3);
    TEST(r.size[1] == 1080);
    TEST(r.size[2] == 1920);

    tensor g = tensor_get(r, 1);
    TEST(g.n == 2);
    TEST(g.size[0] == 1080);
    TEST(g.size[1] == 1920);

    tensor h = tensor_get(g, 34);
    TEST(h.n == 1);
    TEST(h.size[0] == 1920);

    tensor i = tensor_get(h, 13);
    TEST(h.n == 1);
    TEST(h.size[0] == 1920);

    tensor j = tensor_get(i, 0);

    TEST(same_tensor(i, j));

    tensor_free(a);
    tensor_free(b);
    tensor_free(r);
    tensor_free(g);
    tensor_free(h);
    tensor_free(i);
    tensor_free(j);
}

void test_matmul()
{
    {
        tensor m1 = tensor_vmake(2, 2, 2);
        tensor m2 = tensor_vmake(2, 2, 2);
        m1.data[0] = 1;
        m1.data[1] = 2;
        m1.data[2] = 3;
        m1.data[3] = 4;

        m2.data[0] = -1;
        m2.data[1] = -2;
        m2.data[2] = -3;
        m2.data[3] = -4;

        tensor m3 = matrix_multiply(m1, m2);
        tensor tm3 = tensor_vmake(2, 2, 2);
        tm3.data[0] = -7;
        tm3.data[1] = -10;
        tm3.data[2] = -15;
        tm3.data[3] = -22;

        TEST(same_tensor(m3, tm3));

        tensor_free(m1);
        tensor_free(m2);
        tensor_free(m3);
        tensor_free(tm3);
    }
    {
        tensor a = matrix_load("data/test/a.matrix");
        tensor b = matrix_load("data/test/b.matrix");
        tensor c = matrix_load("data/test/c.matrix");
        tensor mul = matrix_multiply(a, b);
        // printf( "A:\n");
        // tensor_print(a);
        // printf( "B:\n");
        // tensor_print(b);
        // printf( "Expected:\n");
        // tensor_print(c);
        // printf( "Received:\n");
        // tensor_print(mul);
        TEST(same_tensor(c, mul));
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
        tensor_free(mul);
    }
}

void test_transpose()
{
    size_t s[2] = {29, 13};
    tensor t = tensor_random(1.0f, 2, s);
    tensor tt = matrix_transpose(t);
    tensor ttt = {0};
    if (tt.n == 2)
    {
        ttt = matrix_transpose(tt);
    }
    TEST (tt.n == 2 && tt.size[0] == t.size[1]);
    TEST (tt.n == 2 && tt.size[1] == t.size[0]);
    TEST (same_tensor(t, ttt));
    tensor_free(t);
    tensor_free(tt);
    tensor_free(ttt);
}

void test_invert()
{
    size_t s[2] = {64, 64};
    tensor t = tensor_random(1.0f, 2, s);
    tensor inv = matrix_invert(t);
    tensor ident = matrix_multiply(t, inv);
    tensor isq = matrix_multiply(ident, ident);
    TEST(same_tensor(ident, isq));
    tensor_free(t);
    tensor_free(inv);
    tensor_free(ident);
    tensor_free(isq);
}

void test_solve_system()
{
    /* Testing solving system of equations:
       a + x - 3 y + z = 2
       -5 a + 3 x - 4 y + z = 0
       a + 2 y - z = 1
       a + 2 x = 12
See: https://www.wolframalpha.com/input/?i=systems+of+equations+calculator&assumption=%22FSelect%22+-%3E+%7B%7B%22SolveSystemOf4EquationsCalculator%22%7D%7D&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation1%22%7D+-%3E%22a+%2B+x+-+3+y+%2B+z+%3D+2%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation2%22%7D+-%3E%22-5+a+%2B+3+x+-+4+y+%2B+z+%3D+0%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation3%22%7D+-%3E%22a+%2B+2+y+-+z+%3D+1%22&assumption=%7B%22F%22%2C+%22SolveSystemOf4EquationsCalculator%22%2C+%22equation4%22%7D+-%3E%22a+%2B+2+x+%3D+12%22
     */

    size_t s[2] = {4, 4};
    size_t sb[2] = {4, 1};
    tensor M = tensor_make(2, s);
    tensor b = tensor_make(2, sb);
    M.data[0] = 1;
    M.data[1] = 1;
    M.data[2] = -3, M.data[3] = 1;
    M.data[4] = -5;
    M.data[5] = 3;
    M.data[6] = -4, M.data[7] = 1;
    M.data[8] = 1;
    M.data[9] = 0;
    M.data[10] = 2, M.data[11] = -1;
    M.data[12] = 1;
    M.data[13] = 2;
    M.data[14] = 0, M.data[15] = 0;

    b.data[0] = 2;
    b.data[1] = 0;
    b.data[2] = 1;
    b.data[3] = 12;
    tensor a = solve_system(M, b);
    tensor t = tensor_make(2, sb);
    t.data[0] = 22. / 17;
    t.data[1] = 91. / 17;
    t.data[2] = 84. / 17;
    t.data[3] = 173. / 17;

    TEST(same_tensor(a, t));

    tensor_free(M);
    tensor_free(b);
    tensor_free(a);
    tensor_free(t);
}

void test_copy()
{
    size_t s[2] = {3, 5};
    tensor t = tensor_random(1.0f, 2, s);
    tensor c = tensor_copy(t);
    TEST(same_tensor(t, c));

    tensor w = tensor_scale(12.3, t);
    TEST(within_eps(w.data[0], t.data[0] * 12.3));
    TEST(within_eps(w.data[11], t.data[11] * 12.3));
    tensor_free(t);
    tensor_free(c);
    tensor_free(w);
}

void test_broadcastable()
{
    tensor t1 = tensor_vmake(4, 11, 5, 13, 9);
    tensor t2 = tensor_vmake(1, 9);
    tensor t3 = tensor_vmake(1, 13);
    tensor t4 = tensor_vmake(2, 1, 9);
    tensor t5 = tensor_vmake(2, 2, 9);
    tensor t6 = tensor_vmake(2, 13, 9);
    tensor t7 = tensor_vmake(4, 11, 1, 13, 1);
    tensor t8 = tensor_vmake(4, 12, 1, 13, 1);
    tensor scalar = tensor_vmake(0);
    TEST(tensor_broadcastable(t1, t2) == 1);
    TEST(tensor_broadcastable(t2, t1) == 1);
    TEST(tensor_broadcastable(t1, t3) == 0);
    TEST(tensor_broadcastable(t3, t1) == 0);
    TEST(tensor_broadcastable(t1, t4) == 1);
    TEST(tensor_broadcastable(t4, t1) == 1);
    TEST(tensor_broadcastable(t1, t5) == 0);
    TEST(tensor_broadcastable(t1, t6) == 1);
    TEST(tensor_broadcastable(t1, t7) == 1);
    TEST(tensor_broadcastable(t1, t8) == 0);
    TEST(tensor_broadcastable(t1, scalar) == 1);
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(t4);
    tensor_free(t5);
    tensor_free(t6);
    tensor_free(t7);
    tensor_free(t8);
}

void test_elementwise()
{
    size_t s1[3] = {3, 2, 5};
    size_t s2[2] = {2, 5};
    size_t s3[1] = {5};
    tensor t1 = tensor_random(1, 3, s1);
    tensor t2 = tensor_random(1, 2, s2);
    tensor t3 = tensor_random(1, 1, s3);
    tensor o23 = tensor_add(t2, t3);

    tensor a2 = tensor_sub(o23, t3);
    // tensor_print(t2);
    // tensor_print(t3);
    // tensor_print(t4);
    // tensor_print(o23);
    // tensor_print(o24);
    tensor scalar = tensor_vmake(0);
    scalar.data[0] = 2;

    tensor t1a = tensor_add(t1, t1);
    tensor t1b = tensor_mul(t1, scalar);
    tensor t1s = tensor_scale(2, t1);
    TEST(same_tensor(t1a, t1s));
    TEST(same_tensor(t1a, t1b));

    TEST(same_tensor(a2, t2));
}

void test_tensor_sum()
{
    size_t s1[3] = {3, 2, 5};
    tensor t1 = tensor_random(1, 3, s1);
    tensor r1 = tensor_sum_dim(t1, 0);
    tensor r2 = tensor_sum_dim(t1, 1);
    tensor r3 = tensor_sum_dim(t1, 2);
    tensor vec = tensor_vrandom(1, 1, 5);
    tensor vecsum = tensor_sum_dim(vec, 0);
    tensor_print(t1);
    tensor_print(r1);
    tensor_print(r2);
    tensor_print(r3);
    tensor_print(vecsum);
}

void time_matrix_multiply()
{
    size_t i;
    size_t n = 100;
    tensor a = tensor_vrandom(1, 2, 512, 768);
    tensor b = tensor_vrandom(1, 2, 768, 384);
    double start = currtime();
    for(i = 0; i < n; ++i){
        tensor c = matrix_multiply(a, b);
        tensor_free(c);
    }
    double end = currtime();
    printf("matrix_multiply took %f sec\n", end - start);
    printf("%g gflops\n", gflops(1.0*n*a.size[0]*b.size[0]*b.size[1], (end-start)));
}

void time_tensor()
{
    {
        size_t s[2] = {512, 512};
        size_t d = sizeof(s) / sizeof(size_t);
        size_t i;
        size_t n = 100;
        tensor a = tensor_random(1, d, s);
        tensor b = tensor_random(1, d, s);
        double start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_add(a, b);
            tensor_free(c);
        }
        double end = currtime();
        printf("tensor_add took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_sub(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_sub took %f sec\n", end - start);
        printf("%g gflops\n", gflops(1.0 * n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_mul(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_mul took %f sec\n", end - start);
        printf("%g gflops\n", gflops(1.0 * n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = matrix_multiply(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("matrix_multiply took %f sec\n", end - start);
        printf("%g gflops\n", gflops(1.0 * n * s[0] * s[1] * s[1], (end - start)));
    }
    {
        size_t s[4] = {512, 128, 2, 2};
        size_t s2[4] = {512, 1, 2, 1};

        size_t i;
        size_t n = 100;
        tensor a = tensor_random(1, 4, s);
        tensor b = tensor_random(1, 4, s2);
        double start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_add(a, b);
            tensor_free(c);
        }
        double end = currtime();
        printf("tensor_add took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_div(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_div took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_mul(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_mul took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));
    }
    {
        size_t s[4] = {512, 128, 2, 2};
        size_t s2[1] = {1};

        size_t i;
        size_t n = 100;
        tensor a = tensor_random(1, 4, s);
        tensor b = tensor_random(1, 1, s2);
        double start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_add(a, b);
            tensor_free(c);
        }
        double end = currtime();
        printf("tensor_add took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_sub(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_sub took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));

        start = currtime();
        for (i = 0; i < n; ++i)
        {
            tensor c = tensor_mul(a, b);
            tensor_free(c);
        }
        end = currtime();
        printf("tensor_mul took %f sec\n", end - start);
        printf("%g gflops\n", gflops(n * s[0] * s[1], (end - start)));
    }
}

void test_activation_layer()
{
    tensor a = matrix_load("data/test/a.matrix");
    tensor truth_alog = matrix_load("data/test/alog.matrix");
    tensor truth_arelu = matrix_load("data/test/arelu.matrix");
    tensor truth_alrelu = matrix_load("data/test/alrelu.matrix");
    tensor truth_asoft = matrix_load("data/test/asoft.matrix");

    layer log_layer = make_activation_layer(LOGISTIC);
    layer relu_layer = make_activation_layer(RELU);
    layer lrelu_layer = make_activation_layer(LRELU);
    layer soft_layer = make_activation_layer(SOFTMAX);

    tensor alog = log_layer.forward(&log_layer, a);
    tensor arelu = relu_layer.forward(&relu_layer, a);
    tensor alrelu = lrelu_layer.forward(&lrelu_layer, a);
    tensor asoft = soft_layer.forward(&soft_layer, a);

    TEST(same_tensor(truth_alog, alog));
    TEST(same_tensor(truth_arelu, arelu));
    TEST(same_tensor(truth_alrelu, alrelu));
    TEST(same_tensor(truth_asoft, asoft));

    tensor y = matrix_load("data/test/y.matrix");
    tensor truth_glog = matrix_load("data/test/glog.matrix");
    tensor truth_grelu = matrix_load("data/test/grelu.matrix");
    tensor truth_glrelu = matrix_load("data/test/glrelu.matrix");
    tensor truth_gsoft = matrix_load("data/test/gsoft.matrix");

    tensor glog = log_layer.backward(&log_layer, y);
    tensor grelu = relu_layer.backward(&relu_layer, y);
    tensor glrelu = lrelu_layer.backward(&lrelu_layer, y);
    tensor gsoft = soft_layer.backward(&soft_layer, y);

    TEST(same_tensor(truth_glog, glog));
    TEST(same_tensor(truth_grelu, grelu));
    TEST(same_tensor(truth_glrelu, glrelu));
    TEST(same_tensor(truth_gsoft, gsoft));

    tensor_free(a);
    tensor_free(y);
    tensor_free(alog);
    tensor_free(arelu);
    tensor_free(alrelu);
    tensor_free(asoft);
    tensor_free(glog);
    tensor_free(grelu);
    tensor_free(glrelu);
    tensor_free(gsoft);
    tensor_free(truth_alog);
    tensor_free(truth_arelu);
    tensor_free(truth_alrelu);
    tensor_free(truth_asoft);
    tensor_free(truth_glog);
    tensor_free(truth_grelu);
    tensor_free(truth_glrelu);
    tensor_free(truth_gsoft);
    free_layer(log_layer);
    free_layer(relu_layer);
    free_layer(lrelu_layer);
    free_layer(soft_layer);
}

void test_connected_layer()
{
    tensor x = matrix_load("data/test/a.matrix");
    tensor w = matrix_load("data/test/b.matrix");
    tensor dw = matrix_load("data/test/dw.matrix");
    tensor db = matrix_load("data/test/db.matrix");
    tensor dy = matrix_load("data/test/dy.matrix");
    tensor truth_dx = matrix_load("data/test/truth_dx.matrix");
    tensor truth_dw = matrix_load("data/test/truth_dw.matrix");
    tensor truth_db = matrix_load("data/test/truth_db.matrix");
    tensor updated_dw = matrix_load("data/test/updated_dw.matrix");
    tensor updated_db = matrix_load("data/test/updated_db.matrix");
    tensor updated_w = matrix_load("data/test/updated_w.matrix");
    tensor updated_b = matrix_load("data/test/updated_b.matrix");

    tensor b = matrix_load("data/test/bias.matrix");
    tensor truth_out = matrix_load("data/test/out.matrix");
    layer l = make_connected_layer(64, 16);
    tensor_free(l.w);
    tensor_free(l.b);
    tensor_free(l.dw);
    tensor_free(l.db);
    l.w = w;
    l.b = b;
    l.dw = dw;
    l.db = db;
    tensor out = l.forward(&l, x);
    TEST(same_tensor(truth_out, out));

    tensor dx = {0};
    if(out.n == 2){
        dx = l.backward(&l, dy);
    }
    TEST(same_tensor(truth_dx, dx));
    TEST(same_tensor(truth_dw, l.dw));
    TEST(same_tensor(truth_db, l.db));

    l.update(&l, 1, .9, .5);
    TEST(same_tensor(updated_dw, l.dw));
    TEST(same_tensor(updated_db, l.db));
    TEST(same_tensor(updated_w, l.w));
    TEST(same_tensor(updated_b, l.b));

    tensor_free(x);
    tensor_free(dx);
    tensor_free(dy);
    tensor_free(out);
    tensor_free(truth_out);
    free_layer(l);
    tensor_free(truth_dx);
    tensor_free(truth_db);
    tensor_free(truth_dw);
    tensor_free(updated_db);
    tensor_free(updated_dw);
    tensor_free(updated_b);
    tensor_free(updated_w);
}

void test_mattrans()
{
    /* Make tensor A */
    size_t s[2] = {3, 2};
    tensor a = tensor_make(2, s);
    for (int i = 0; i < 6; i++)
    {
        a.data[i] = i;
    }
    printf("Original A:\n");
    tensor_print(a);

    /* Transform tensor A */
    tensor a_t = matrix_transpose(a);
    printf("Transpose A:\n");
    tensor_print(a_t);
}

void test_matmul_small()
{
    /* Make tensor A */
    size_t a_n = 2;
    size_t a_m = 4;
    size_t s[2] = {a_n, a_m};
    tensor a = tensor_make(2, s);
    for (int i = 0; i < a_n * a_m; i++)
    {
        a.data[i] = i;
    }

    /* Make tensor B */
    size_t b_n = 4;
    size_t b_m = 1;
    size_t s2[2] = {b_n, b_m};
    tensor b = tensor_make(2, s2);
    for (int i = 0; i < b_n * b_m; i++)
    {
        b.data[i] = i;
    }

    printf("Tensor A:\n");
    tensor_print(a);
    printf("Tensor B:\n");
    tensor_print(b);

    tensor c = matrix_multiply(a, b);

    printf("Tensor A*B:\n");
    tensor_print(c);
}

void test_forward_activation_layer()
{
    tensor a = matrix_load("data/test/a.matrix");

    layer log_layer = make_activation_layer(LOGISTIC);
    layer relu_layer = make_activation_layer(RELU);
    layer lrelu_layer = make_activation_layer(LRELU);
    layer soft_layer = make_activation_layer(SOFTMAX);

    tensor b = log_layer.forward(&log_layer, a);
}

void test_im2col_small() {
    size_t c = 2, h = 2, w = 2;
    size_t s[3] = {c, h, w};
    tensor a = tensor_make(3, s);
    for (int i = 0; i < c * h * w; i++)
    {
        a.data[i] = i;
    }
    printf("Original A:\n");
    tensor_print(a);
    tensor col = im2col(a, 2, 2, 1, 0);
    printf("Col A:\n");
    tensor_print(col);
}

void test_im2col()
{
    image im = load_image("data/test/dog.jpg"); 
    tensor t = image_to_tensor(im);
    tensor col = im2col(t, 3, 3, 2, 1);
    tensor truth_col = matrix_load("data/test/im2col.matrix");
    tensor col2 = im2col(t, 2, 2, 2, 0);
    tensor truth_col2 = matrix_load("data/test/im2col2.matrix");
    TEST(same_tensor(truth_col,   col));
    TEST(same_tensor(truth_col2,  col2));
    tensor_free(col);
    tensor_free(col2);
    tensor_free(truth_col);
    tensor_free(truth_col2);
    tensor_free(t);
    free_image(im);
}

void test_col2im()
{
    image im = load_image("data/test/dog.jpg"); 
    tensor dcol = matrix_load("data/test/dcol.matrix");
    tensor dcol2 = matrix_load("data/test/dcol2.matrix");
    tensor col2im_res = col2im(dcol, im.c, im.h, im.w, 3, 3, 2, 1);
    tensor col2im_res2 = col2im(dcol2, im.c, im.h, im.w, 2, 2, 2, 0);

    tensor truth_col2im = tensor_load("data/test/truth_col2im.tensor");
    tensor truth_col2im2 = tensor_load("data/test/truth_col2im2.tensor");

    TEST(same_tensor(truth_col2im, col2im_res));
    TEST(same_tensor(truth_col2im2, col2im_res2));
    free_image(im);
}


void test_convolutional_layer()
{
    layer l = make_convolutional_layer(16, 8, 3, 1, 1);
    tensor xt = tensor_load("data/test/conv_x.tensor");
    tensor wt = tensor_load("data/test/conv_w.tensor");
    tensor bt = tensor_load("data/test/conv_b.tensor");
    tensor truth_yt = tensor_load("data/test/conv_y.tensor");

    tensor_free(l.w);
    tensor_free(l.b);
    l.w = wt;
    l.b = bt;

    tensor y = l.forward(&l, xt);
 
    TEST(same_tensor(truth_yt, y));

    tensor dyt = tensor_load("data/test/conv_dy.tensor");
    tensor dwt = tensor_load("data/test/conv_dw.tensor");
    tensor dbt = tensor_load("data/test/conv_db.tensor");
    tensor truth_dxt = tensor_load("data/test/conv_dx.tensor");
    
    tensor dx = l.backward(&l, dyt);

    TEST(same_tensor(truth_dxt, dx));
    TEST(same_tensor(dwt, l.dw));
    TEST(same_tensor(dbt, l.db));

    l.update(&l, 1, .9, .5);

    tensor updated_wt =  tensor_load("data/test/updated_conv_w.tensor");
    tensor updated_dwt = tensor_load("data/test/updated_conv_dw.tensor");
    tensor updated_bt =  tensor_load("data/test/updated_conv_b.tensor");
    tensor updated_dbt = tensor_load("data/test/updated_conv_db.tensor");

    TEST(same_tensor(updated_dwt, l.dw));
    TEST(same_tensor(updated_wt, l.w));
    TEST(same_tensor(updated_dbt, l.db));
    TEST(same_tensor(updated_bt, l.b));

    tensor_free(updated_wt);
    tensor_free(updated_dwt);
    tensor_free(updated_bt);
    tensor_free(updated_dbt);

    tensor_free(dyt);
    tensor_free(dbt);
    tensor_free(dwt);

    free_layer(l);
    tensor_free(truth_yt);
    tensor_free(xt);
    tensor_free(dx);
    tensor_free(y);
}

void test_maxpool_layer()
{
    tensor xt = tensor_load("data/test/max_x.tensor");

    layer max_l = make_maxpool_layer(2, 2);
    layer max_l3 = make_maxpool_layer(3, 2);

    tensor max_y = max_l.forward(&max_l, xt);
    tensor max_y3 = max_l3.forward(&max_l3, xt);

    tensor truth_max_yt =  tensor_load("data/test/max_y.tensor");
    tensor truth_max_y3t = tensor_load("data/test/max_y3.tensor");

    TEST(same_tensor(truth_max_yt, max_y));
    TEST(same_tensor(truth_max_y3t, max_y3));

    tensor max_dyt =  tensor_load("data/test/max_dy.tensor");
    tensor max_dy3t = tensor_load("data/test/max_dy3.tensor");

    tensor max_dx = max_l.backward(&max_l, max_dyt);
    tensor max_dx3 = max_l3.backward(&max_l3, max_dy3t);

    tensor truth_max_dxt =  tensor_load("data/test/max_dx.tensor");
    tensor truth_max_dx3t = tensor_load("data/test/max_dx3.tensor");
    

    TEST(same_tensor(truth_max_dxt, max_dx));
    TEST(same_tensor(truth_max_dx3t, max_dx3));


    tensor_free(xt);
    tensor_free(max_y);
    tensor_free(max_y3);
    tensor_free(truth_max_yt);
    tensor_free(truth_max_y3t);
    tensor_free(max_dx);
    tensor_free(max_dx3);
    tensor_free(truth_max_dxt);
    tensor_free(truth_max_dx3t);
    free_layer(max_l);
    free_layer(max_l3);
}

void test_hw0()
{
    // custom tests
    // test_matmul_small();
    // test_mattrans();
    // test_forward_activation_layer();

    // provided tests
    test_copy();
    test_transpose();
    test_matmul();
    test_activation_layer();
    test_connected_layer();
}

void test_hw1()
{
    // test_im2col_small();
    test_im2col();
    test_col2im();
    test_convolutional_layer();
    test_maxpool_layer();
}

void test()
{
    test_tensor_make_get();
    test_transpose();
    test_invert();
    test_solve_system();
    test_broadcastable();
    test_elementwise();
    test_tensor_sum();
    time_tensor();
    // printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

