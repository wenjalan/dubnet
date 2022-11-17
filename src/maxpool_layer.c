#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    tensor y = tensor_vmake(4,
                            x.size[0], // same # data points and # of channels (N and C)
                            x.size[1],
                            (x.size[2] - 1) / l->stride + 1, // H and W scaled based on stride
                            (x.size[3] - 1) / l->stride + 1);

    // This might be a useful offset...
    int pad = -((int)l->size - 1) / 2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    // for each region
    for (size_t n = 0; n < x.size[0]; n++)
    {
        // for each channel
        for (size_t c = 0; c < x.size[1]; c++)
        {
            // for each row
            for (size_t h = 0; h < y.size[2]; h++)
            {
                // for each col
                for (size_t w = 0; w < y.size[3]; w++)
                {
                    // find max value in region
                    // this some black magic
                    float max = -FLT_MAX;
                    for (size_t i = 0; i < l->size; i++)
                    {
                        for (size_t j = 0; j < l->size; j++)
                        {
                            size_t x_h = h * l->stride + i + pad;
                            size_t x_w = w * l->stride + j + pad;
                            if (x_h >= 0 && x_h < x.size[2] && x_w >= 0 && x_w < x.size[3])
                            {
                                float val = x.data[n * x.size[1] * x.size[2] * x.size[3] + c * x.size[2] * x.size[3] + x_h * x.size[3] + x_w];
                                if (val > max)
                                {
                                    max = val;
                                }
                            }
                        }
                    }
                    // set output to max value
                    y.data[n * y.size[1] * y.size[2] * y.size[3] + c * y.size[2] * y.size[3] + h * y.size[3] + w] = max;
                }
            }
        }
    }
    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int)l->size - 1) / 2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (size_t n = 0; n < x.size[0]; n++)
    {
        // for each channel
        for (size_t c = 0; c < x.size[1]; c++)
        {
            // for each row
            for (size_t h = 0; h < dy.size[2]; h++)
            {
                // for each col
                for (size_t w = 0; w < dy.size[3]; w++)
                {
                    // find max value in region
                    // this some more black magic
                    float max = -FLT_MAX;
                    size_t max_i = 0;
                    size_t max_j = 0;
                    for (size_t i = 0; i < l->size; i++)
                    {
                        for (size_t j = 0; j < l->size; j++)
                        {
                            size_t x_h = h * l->stride + i + pad;
                            size_t x_w = w * l->stride + j + pad;
                            if (x_h >= 0 && x_h < x.size[2] && x_w >= 0 && x_w < x.size[3])
                            {
                                float val = x.data[n * x.size[1] * x.size[2] * x.size[3] + c * x.size[2] * x.size[3] + x_h * x.size[3] + x_w];
                                if (val > max)
                                {
                                    max = val;
                                    max_i = i;
                                    max_j = j;
                                }
                            }
                        }
                    }
                    // set output to max value
                    dx.data[n * dx.size[1] * dx.size[2] * dx.size[3] + c * dx.size[2] * dx.size[3] + (h * l->stride + max_i + pad) * dx.size[3] + (w * l->stride + max_j + pad)] = dy.data[n * dy.size[1] * dy.size[2] * dy.size[3] + c * dy.size[2] * dy.size[3] + h * dy.size[3] + w];
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay) {}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update = update_maxpool_layer;
    return l;
}
