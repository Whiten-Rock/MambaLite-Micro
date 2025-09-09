/**
 * @file mamba.c
 * @brief mamba inference kernel
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once
#include "matrix.h"
#include "mamba.h"
#include "tensor.c"
#include "mamba_weights.c"

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
float silu(float x)
{
    return x * sigmoid(x);
}
float softplus(float x)
{
    return logf(1.0f + expf(x));
    // if (x > 20.0f)
    //     return x;
    // else if (x < -20.0f)
    //     return expf(x);
    // else
    //     return logf(1.0f + expf(x));
}
typedef struct
{
    uint16_t d_model;
    uint16_t d_state;
    uint16_t d_conv;
    uint16_t expand;
    uint16_t d_inner;
    uint16_t dt_rank;
    Matrix *in_proj_weight;
    Matrix *A_log;
    Matrix *conv1d_weight;
    float *conv1d_bias;
    Matrix *x_proj_weight;
    float *dt_proj_bias;
    Matrix *dt_proj_weight;
    float *D;
    Matrix *out_proj_weight;
} Mamba_Variables;

Mamba_Variables *Mamba_Init()
{
    Mamba_Variables *m = (Mamba_Variables *)malloc(sizeof(Mamba_Variables));
    m->d_model = mamba_d_model;
    m->d_state = mamba_d_state;
    m->d_conv = mamba_d_conv;
    m->d_inner = mamba_expand * mamba_d_model;
    m->dt_rank = mamba_dt_rank;
    m->in_proj_weight = &mamba_in_proj_weight_matrix;
    m->A_log = &mamba_A_log_matrix;
    m->conv1d_weight = &mamba_conv1d_weight_matrix;
    m->conv1d_bias = (float *)mamba_conv1d_bias;
    m->x_proj_weight = &mamba_x_proj_weight_matrix;
    m->dt_proj_weight = &mamba_dt_proj_weight_matrix;
    m->dt_proj_bias = (float *)mamba_dt_proj_bias;
    m->D = (float *)mamba_D;
    m->out_proj_weight = &mamba_out_proj_weight_matrix;
    return m;
}

MambaError Mamba_Forward(Mamba_Variables *self, Tensor *hidden_states, Tensor *out)
{
    // batch, seqlen, dim = hidden_states.shape
    int batch = hidden_states->d1;
    int seqlen = hidden_states->d2;
    int dim = hidden_states->d3;
    int b = hidden_states->d1;
    int l = hidden_states->d2;
    int d = hidden_states->d3;
    /*xz = rearrange(
        self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )*/
    Matrix *xz = matrix_zeros(self->d_inner, b * l);
    if (!xz) return Mamba_ERR_MEMORY;
    for (int bi = 0; bi < b; bi++)
    {
        for (int li = 0; li < l; li++)
        {
            for (int di = 0; di < self->d_inner; di++)
            {
                int in_idx = bi * l * d + li * d + di;
                int out_idx = di * (b * l) + (bi * l + li);
                xz->data[out_idx] = hidden_states->data[in_idx];
            }
        }
    }
    Matrix *result = matrix_zeros(self->d_inner * 2, b * l);
    if (!result) return Mamba_ERR_MEMORY;
    for (int i = 0; i < self->d_inner * 2; i++) // rows of weight
    {
        for (int j = 0; j < b * l; j++) // columns of xz
        {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) // inner dimension
            {
                int w_idx = i * d + k;        // weight[i][k]
                int xz_idx = k * (b * l) + j; // xz[k][j]
                sum += self->in_proj_weight->data[w_idx] * xz->data[xz_idx];
            }
            int out_idx = i * (b * l) + j; // result[i][j]
            result->data[out_idx] = sum;
        }
    }
    matrix_free(xz);
    // x, z = xz.chunk(2, dim=1)
    Tensor *x = tensor_zeros(b, self->d_inner, l);
    if (!x) return Mamba_ERR_MEMORY;
    Tensor *z = tensor_zeros(b, self->d_inner, l);
    if (!z) return Mamba_ERR_MEMORY;
    for (int di = 0; di < self->d_inner; di++)
    {
        for (int bi = 0; bi < b; bi++)
        {
            for (int li = 0; li < l; li++)
            {
                int out_idx = bi * self->d_inner * l + di * l + li;
                int in_idx = di * (b * l) + bi * l + li;
                x->data[out_idx] = result->data[in_idx];
            }
        }
    }
    for (int di = self->d_inner; di < 2 * self->d_inner; di++)
    {
        int dz = di - self->d_inner;
        for (int bi = 0; bi < b; bi++)
        {
            for (int li = 0; li < l; li++)
            {
                int out_idx = bi * self->d_inner * l + dz * l + li;
                int in_idx = di * (b * l) + bi * l + li;
                z->data[out_idx] = result->data[in_idx];
            }
        }
    }
    matrix_free(result);
    // x = self.act(self.conv1d(x)[..., :seqlen])
    int kernel_size = self->d_conv;
    int padding = kernel_size - 1;
    int padded_len = seqlen + 2 * padding;
    int groups = self->d_inner;
    float *padded = (float *) calloc(padded_len, sizeof(float));
    if (!padded) return Mamba_ERR_MEMORY;
    for (int b = 0; b < batch; b++)
    {
        for (int c = 0; c < groups; c++)
        {
            memset(padded, 0, sizeof(float) * padded_len);
            for (int i = 0; i < seqlen; i++)
            {
                padded[i + padding] = x->data[b * groups * seqlen + c * seqlen + i];
            }
            for (int i = 0; i < seqlen; i++)
            {
                float sum = 0.0f;
                for (int k = 0; k < kernel_size; k++)
                {
                    sum += padded[i + k] * self->conv1d_weight->data[c * kernel_size + k];
                }
                sum += self->conv1d_bias[c];
                float activated = silu(sum);
                x->data[b * groups * seqlen + c * seqlen + i] = activated;
            }
        }
    }
    free(padded);
    // x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
    Matrix *x_dbl = matrix_zeros(b * l, self->dt_rank + 2 * self->d_state);
    if (!x_dbl) return Mamba_ERR_MEMORY;
    Matrix *x_bld = matrix_zeros(b * l, self->d_inner);
    if (!x_bld) return Mamba_ERR_MEMORY;
    for (int bi = 0; bi < b; bi++)
    {
        for (int di = 0; di < self->d_inner; di++)
        {
            for (int li = 0; li < l; li++)
            {
                int in_idx = bi * self->d_inner * l + di * l + li; // x[bi][di][li]
                int out_idx = (bi * l + li) * self->d_inner + di;  // xz[bi * l + li][di]
                x_bld->data[out_idx] = x->data[in_idx];
            }
        }
    }
    int out_dim = self->dt_rank + self->d_state * 2;
    for (int bi = 0; bi < b * l; bi++)
    {
        for (int oi = 0; oi < out_dim; ++oi)
        {
            float sum = 0.0f;
            for (int i = 0; i < self->d_inner; i++)
            {
                sum += x_bld->data[bi * self->d_inner + i] * self->x_proj_weight->data[oi * self->d_inner + i];
            }
            x_dbl->data[bi * out_dim + oi] = sum;
        }
    }
    matrix_free(x_bld);
    // dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    Matrix *dt_T = matrix_zeros(b * l, self->dt_rank);
    if (!dt_T) return Mamba_ERR_MEMORY;
    Matrix *B_matrix = matrix_zeros(b * l, self->d_state);
    if (!B_matrix) return Mamba_ERR_MEMORY;
    Matrix *C_matrix = matrix_zeros(b * l, self->d_state);
    if (!C_matrix) return Mamba_ERR_MEMORY;
    int total_dim = self->dt_rank + 2 * self->d_state;
    for (int i = 0; i < b * l; ++i)
    {
        for (int j = 0; j < self->dt_rank; ++j)
        {
            dt_T->data[i * self->dt_rank + j] = x_dbl->data[i * total_dim + j];
        }
        for (int j = 0; j < self->d_state; ++j)
        {
            B_matrix->data[i * self->d_state + j] = x_dbl->data[i * total_dim + self->dt_rank + j];

            C_matrix->data[i * self->d_state + j] = x_dbl->data[i * total_dim + self->dt_rank + self->d_state + j];
        }
    }
    matrix_free(x_dbl);
    // dt = self.dt_proj.weight @ dt.t()
    Matrix *dt_matrix = matrix_zeros(b * l, self->d_inner);
    if (!dt_matrix) return Mamba_ERR_MEMORY;
    for (int i = 0; i < b * l; ++i)
    {
        for (int j = 0; j < self->d_inner; ++j)
        {
            float sum = 0;
            for (int k = 0; k < self->dt_rank; ++k)
            {
                sum += self->dt_proj_weight->data[j * self->dt_rank + k] * dt_T->data[i * self->dt_rank + k];
            }
            dt_matrix->data[i * self->d_inner + j] = sum;
        }
    }
    matrix_free(dt_T);
    // dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    // B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    // C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    Tensor *dt = tensor_zeros(b, self->d_inner, l);
    if (!dt) return Mamba_ERR_MEMORY;
    Tensor *B = tensor_zeros(b, self->d_state, l);
    if (!B) return Mamba_ERR_MEMORY;
    Tensor *C = tensor_zeros(b, self->d_state, l);
    if (!C) return Mamba_ERR_MEMORY;
    for (int bi = 0; bi < b; ++bi)
    {
        for (int li = 0; li < l; ++li)
        {
            int row_idx = bi * l + li;

            for (int di = 0; di < self->d_inner; ++di)
            {
                int dst_idx = bi * self->d_inner * l + di * l + li;
                dt->data[dst_idx] = dt_matrix->data[row_idx * self->d_inner + di];
            }

            for (int di = 0; di < self->d_state; ++di)
            {
                int dst_idx = bi * self->d_state * l + di * l + li;
                B->data[dst_idx] = B_matrix->data[row_idx * self->d_state + di];
                C->data[dst_idx] = C_matrix->data[row_idx * self->d_state + di];
            }
        }
    }
    matrix_free(dt_matrix);
    matrix_free(B_matrix);
    matrix_free(C_matrix);
    // delta = delta + delta_bias[..., None].float()
    for (int bi = 0; bi < b; ++bi)
    {
        for (int di = 0; di < self->d_inner; ++di)
        {
            float bias = self->dt_proj_bias[di];
            for (int li = 0; li < l; ++li)
            {
                int idx = bi * self->d_inner * l + di * l + li;
                dt->data[idx] += bias;
            }
        }
    }
    // delta = F.softplus(delta)
    for (int bi = 0; bi < b; ++bi)
    {
        for (int di = 0; di < self->d_inner; ++di)
        {
            for (int li = 0; li < l; ++li)
            {
                int idx = bi * self->d_inner * l + di * l + li;
                dt->data[idx] = softplus(dt->data[idx]);
            }
        }
    }
    //
    // x = A.new_zeros((batch, dim, dstate))
    Tensor *x_ssm = tensor_zeros(batch, self->d_inner, self->d_state);
    if (!x_ssm) return Mamba_ERR_MEMORY;
    Tensor *y_out = tensor_zeros(batch, self->d_inner, seqlen);
    if (!y_out) return Mamba_ERR_MEMORY;
    d = self->d_inner;
    // A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    Matrix *A = matrix_zeros(self->d_inner, self->d_state);
    if (!A) return Mamba_ERR_MEMORY;
    for (int i = 0; i < self->d_inner; i++)
    {
        for (int j = 0; j < self->d_state; j++)
        {
            int idx = i * self->d_state + j;
            A->data[idx] = -expf(self->A_log->data[idx]); // float precision exponent
        }
    }
    for (int bi = 0; bi < b; ++bi)
    {
        for (int di = 0; di < d; ++di)
        {
            for (int t = 0; t < l; ++t)
            {
                float delta_val = dt->data[bi * d * l + di * l + t];
                float u_val = x->data[bi * d * l + di * l + t];
                for (int ni = 0; ni < self->d_state; ++ni)
                {
                    float A_val = A->data[di * self->d_state + ni];
                    float B_val = B->data[bi * self->d_state * l + ni * l + t];
                    float C_val = C->data[bi * self->d_state * l + ni * l + t];

                    float decay = expf(delta_val * A_val);
                    int x_idx = (bi * d + di) * self->d_state + ni;
                    x_ssm->data[x_idx] = decay * x_ssm->data[x_idx] + delta_val * B_val * u_val;

                    int y_idx = (bi * d + di) * l + t;
                    y_out->data[y_idx] += x_ssm->data[x_idx] * C_val;
                }
            }
        }
    }
    tensor_free(dt);
    tensor_free(x_ssm);
    tensor_free(B);
    tensor_free(C);
    matrix_free(A);
    for (int bi = 0; bi < b; ++bi)
    {
        for (int di = 0; di < self->d_inner; ++di)
        {
            float D_val = self->D[di];
            for (int t = 0; t < l; ++t)
            {
                float u_val = x->data[bi * self->d_inner * l + di * l + t];
                float z_val = z->data[bi * self->d_inner * l + di * l + t];
                float z_act = silu(z_val);
                int idx = (bi * self->d_inner + di) * l + t;
                y_out->data[idx] = (y_out->data[idx] + u_val * D_val) * z_act;
            }
        }
    }
    tensor_free(x);
    tensor_free(z);
    // y = rearrange(y, "b d l -> b l d")
    Tensor *y = tensor_zeros(batch, seqlen, self->d_inner);
    if (!y) return Mamba_ERR_MEMORY;
    for (int b = 0; b < batch; ++b)
    {
        for (int d = 0; d < self->d_inner; ++d)
        {
            for (int l = 0; l < seqlen; ++l)
            {
                int src_idx = (b * self->d_inner + d) * seqlen + l; // y_out[b, d, l]
                int dst_idx = (b * seqlen + l) * self->d_inner + d; // y[b, l, d]
                y->data[dst_idx] = y_out->data[src_idx];
            }
        }
    }
    tensor_free(y_out);
    // out = self.out_proj(y)
    for (int b = 0; b < batch; ++b)
    {
        for (int l = 0; l < seqlen; ++l)
        {
            for (int j = 0; j < self->d_model; ++j) // output dim
            {
                float sum = 0.0f;
                for (int i = 0; i < self->d_inner; ++i) // input dim
                {
                    int y_idx = (b * seqlen + l) * self->d_inner + i; // y[b, l, i]
                    int w_idx = j * self->d_inner + i;                // W[j, i]
                    sum += y->data[y_idx] * self->out_proj_weight->data[w_idx];
                }

                int out_idx = (b * seqlen + l) * self->d_model + j; // out[b, l, j]
                out->data[out_idx] = sum;
            }
        }
    }
    tensor_free(y);
    return Mamba_OK;
}