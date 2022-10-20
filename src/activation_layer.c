#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"

// x = x
float linear_activation(float x)
{
    return x;
}

// logistic(x) = 1/(1+e^(-x))
float logistic_activation(float x)
{
    return 1 / (1 + pow(M_E, (-x)));
}

// relu(x)     = x if x > 0 else 0
float relu_activation(float x)
{
    return x > 0 ? x : 0;
}

// lrelu(x)    = x if x > 0 else .01 * x
float leaky_relu_activation(float x)
{
    return x > 0 ? x : 0.01 * x;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer y = f(x)
tensor forward_activation_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    ACTIVATION a = l->activation;
    tensor y = tensor_copy(x);

    // TODO: 2.0
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row

    assert(x.n >= 2);

    float (*activation)(float);
    switch (a)
    {
    case LINEAR:
        activation = linear_activation;
        break;
    case LOGISTIC:
        activation = logistic_activation;
        break;
    case RELU:
        activation = relu_activation;
        break;
    case LRELU:
        activation = leaky_relu_activation;
        break;
    }

    /* You might want this */
    size_t i, j;
    for (i = 0; i < x.size[0]; ++i)
    {
        tensor x_i = tensor_get_(x, i);
        tensor y_i = tensor_get_(y, i);
        size_t len = tensor_len(x_i);

        float softmax_sum = 0.0f;
        if (a == SOFTMAX) {
            for (j = 0; j < len; ++j) {
                softmax_sum += pow(M_E, x_i.data[j]);
            }
        }

        // Do stuff in here
        for (j = 0; j < len; ++j)
        {
            // TODO: Implement activation functions
            if (a == SOFTMAX)
            {
                // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row
                y_i.data[j] = pow(M_E, x_i.data[j]) / softmax_sum;
            }
            else
            {
                y_i.data[j] = activation(x_i.data[j]);
            }
        }
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
tensor backward_activation_layer(layer *l, tensor dy)
{
    tensor x = l->x;
    tensor dx = tensor_copy(dy);
    ACTIVATION a = l->activation;

    // TODO: 2.1
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    /* Might want this too
    size_t i, j;
    for(i = 0; i < dx.size[0]; ++i){
        tensor x_i = tensor_get_(x, i);
        tensor dx_i = tensor_get_(dx, i);
        size_t len = tensor_len(dx_i);
        // Do stuff in here
    }
    */

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer *l, float rate, float momentum, float decay) {}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
