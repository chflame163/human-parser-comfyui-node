#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include <vector>

#include "utils/checks.h"
#include "utils/cuda.cuh"
#include "inplace_abn.h"

#include <ATen/cuda/CUDAContext.h>

// Operations for reduce
struct SumOpH {
  __device__ SumOpH(const half *t, int c, int s)
      : tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    return __half2float(tensor[(batch * chn + plane) * sp + n]);
  }
  const half *tensor;
  const int chn;
  const int sp;
};

struct VarOpH {
  __device__ VarOpH(float m, const half *t, int c, int s)
      : mean(m), tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    const auto t = __half2float(tensor[(batch * chn + plane) * sp + n]);
    return (t - mean) * (t - mean);
  }
  const float mean;
  const half *tensor;
  const int chn;
  const int sp;
};

struct GradOpH {
  __device__ GradOpH(float _weight, float _bias, const half *_z, const half *_dz, int c, int s)
      : weight(_weight), bias(_bias), z(_z), dz(_dz), chn(c), sp(s) {}
  __device__ __forceinline__ Pair<float> operator()(int batch, int plane, int n) {
    float _y = (__half2float(z[(batch * chn + plane) * sp + n]) - bias) / weight;
    float _dz = __half2float(dz[(batch * chn + plane) * sp + n]);
    return Pair<float>(_dz, _y * _dz);
  }
  const float weight;
  const float bias;
  const half *z;
  const half *dz;
  const int chn;
  const int sp;
};

// 添加类型转换辅助函数
template<typename T>
__device__ __forceinline__ half to_half(T x) {
    return __float2half(static_cast<float>(x));
}

template<typename T>
__device__ __forceinline__ float from_half(T x) {
    return __half2float(x);
}

/***********
 * mean_var
 ***********/

__global__ void mean_var_kernel_h(const c10::Half *x, float *mean, float *var, int num, int chn, int sp) {
  int plane = blockIdx.x;
  float norm = 1.f / static_cast<float>(num * sp);

  float _mean = reduce<float, SumOpH>(SumOpH(reinterpret_cast<const half*>(x), chn, sp), plane, num, sp) * norm;
  __syncthreads();
  float _var = reduce<float, VarOpH>(VarOpH(_mean, reinterpret_cast<const half*>(x), chn, sp), plane, num, sp) * norm;

  if (threadIdx.x == 0) {
    mean[plane] = _mean;
    var[plane] = _var;
  }
}

std::vector<at::Tensor> mean_var_cuda_h(at::Tensor x) {
  CHECK_CUDA_INPUT(x);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto mean = at::empty({chn},x.options().dtype(at::kFloat));
  auto var = at::empty({chn},x.options().dtype(at::kFloat));

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  mean_var_kernel_h<<<blocks, threads, 0, stream>>>(
      x.data_ptr<c10::Half>(),
      mean.data_ptr<float>(),
      var.data_ptr<float>(),
      num, chn, sp);

  return {mean, var};
}

/**********
 * forward
 **********/

__global__ void forward_kernel_h(c10::Half *x, const float *mean, const float *var, const float *weight, const float *bias,
                                 bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  const float _mean = mean[plane];
  const float _var = var[plane];
  const float _weight = affine ? abs(weight[plane]) + eps : 1.f;
  const float _bias = affine ? bias[plane] : 0.f;

  const float mul = rsqrt(_var + eps) * _weight;

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      c10::Half *x_ptr = x + (batch * chn + plane) * sp + n;
      float _x = __half2float(*reinterpret_cast<half*>(x_ptr));
      float _y = (_x - _mean) * mul + _bias;

      *reinterpret_cast<half*>(x_ptr) = __float2half(_y);
    }
  }
}

at::Tensor forward_cuda_h(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps) {
  CHECK_CUDA_INPUT(x);
  CHECK_CUDA_INPUT(mean);
  CHECK_CUDA_INPUT(var);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  forward_kernel_h<<<blocks, threads, 0, stream>>>(
      x.data_ptr<c10::Half>(),
      mean.data_ptr<float>(),
      var.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      affine, eps, num, chn, sp);

  return x;
}

__global__ void edz_eydz_kernel_h(const c10::Half *z, const c10::Half *dz, const float *weight, const float *bias,
                                float *edz, float *eydz, bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  float _weight = affine ? abs(weight[plane]) + eps : 1.f;
  float _bias = affine ? bias[plane] : 0.f;

  Pair<float> res = reduce<Pair<float>, GradOpH>(GradOpH(_weight, _bias, 
    reinterpret_cast<const half*>(z), 
    reinterpret_cast<const half*>(dz), chn, sp), plane, num, sp);
  __syncthreads();

  if (threadIdx.x == 0) {
    edz[plane] = res.v1;
    eydz[plane] = res.v2;
  }
}

std::vector<at::Tensor> edz_eydz_cuda_h(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                      bool affine, float eps) {
  CHECK_CUDA_INPUT(z);
  CHECK_CUDA_INPUT(dz);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto edz = at::empty({chn},z.options().dtype(at::kFloat));
  auto eydz = at::empty({chn},z.options().dtype(at::kFloat));

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  edz_eydz_kernel_h<<<blocks, threads, 0, stream>>>(
        z.data_ptr<at::Half>(),
        dz.data_ptr<at::Half>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        edz.data_ptr<float>(),
        eydz.data_ptr<float>(),
        affine, eps, num, chn, sp);
 
  return {edz, eydz};
}

__global__ void backward_kernel_h(const c10::Half *z, const c10::Half *dz, const float *var, const float *weight, const float *bias, 
                                const float *edz, const float *eydz, c10::Half *dx, bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  float _weight = affine ? abs(weight[plane]) + eps : 1.f;
  float _bias = affine ? bias[plane] : 0.f;
  float _var = var[plane];
  float _edz = edz[plane];
  float _eydz = eydz[plane];

  float _mul = _weight * rsqrt(_var + eps);
  float count = float(num * sp);

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      float _dz = __half2float(*reinterpret_cast<const half*>(&dz[(batch * chn + plane) * sp + n]));
      float _y = (__half2float(*reinterpret_cast<const half*>(&z[(batch * chn + plane) * sp + n])) - _bias) / _weight;

      *reinterpret_cast<half*>(&dx[(batch * chn + plane) * sp + n]) = __float2half((_dz - _edz / count - _y * _eydz / count) * _mul);
    }
  }
}

at::Tensor backward_cuda_h(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                      at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  CHECK_CUDA_INPUT(z);
  CHECK_CUDA_INPUT(dz);
  CHECK_CUDA_INPUT(var);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);
  CHECK_CUDA_INPUT(edz);
  CHECK_CUDA_INPUT(eydz);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto dx = at::zeros_like(z);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  backward_kernel_h<<<blocks, threads, 0, stream>>>(
        z.data_ptr<at::Half>(),
        dz.data_ptr<at::Half>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        edz.data_ptr<float>(),
        eydz.data_ptr<float>(),
        dx.data_ptr<at::Half>(),
        affine, eps, num, chn, sp);

  return dx;
}

__global__ void leaky_relu_backward_impl_h(c10::Half *z, c10::Half *dz, float slope, int64_t count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;  i += blockDim.x * gridDim.x){
    float _z = __half2float(*reinterpret_cast<half*>(&z[i]));
    if (_z < 0) {
      *reinterpret_cast<half*>(&dz[i]) = __float2half(__half2float(*reinterpret_cast<half*>(&dz[i])) * slope);
      *reinterpret_cast<half*>(&z[i]) = __float2half(_z / slope);
    }
  }
}

void leaky_relu_backward_cuda_h(at::Tensor z, at::Tensor dz, float slope) {
  CHECK_CUDA_INPUT(z);
  CHECK_CUDA_INPUT(dz);

  int64_t count = z.numel();
  dim3 threads(getNumThreads(count));
  dim3 blocks = (count + threads.x - 1) / threads.x;
  auto stream = at::cuda::getCurrentCUDAStream();
  leaky_relu_backward_impl_h<<<blocks, threads, 0, stream>>>(
      z.data_ptr<at::Half>(),
      dz.data_ptr<at::Half>(),
      slope, count);
}

