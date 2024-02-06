import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

img = io.read_image('puppy.jpg')

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
}

__global__ void rgb_to_grayscale_kernel_float(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = 0.2989f*x[i] + 0.5870f*x[i+n] + 0.1140f*x[i+2*n];
}

__global__ void rgb_to_grayscale_kernel_int(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = ((77*x[i] + 150*x[i+n] + 29*x[i+2*n]) >> 8);
}

__global__ void rgb_to_grayscale_kernel_float_coalesced(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>n) return;
    
    unsigned int *xUint = (unsigned int*)x;
    const unsigned int rVec = xUint[i];
    const unsigned int gVec = xUint[i+n];
    const unsigned int bVec = xUint[i+2*n];
    unsigned char greyVec[4];
    for (int i = 0; i < 4; i++) {
        const unsigned char r = ((rVec >> i * 8) & 0xFFU);
        const unsigned char g = ((gVec >> i * 8) & 0xFFU);
        const unsigned char b = ((bVec >> i * 8) & 0xFFU);
        greyVec[i] = 0.2989f*r + 0.5870f*g + 0.1140f*b;
    }

    unsigned int *outUint = (unsigned int*)out;
    outUint[i] = *((unsigned int*)greyVec);
}

torch::Tensor rgb_to_grayscale(torch::Tensor input, int approach, bool internalTimer) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty({h,w}, input.options());
    int threads = 256;
    cudaEvent_t start, end;
    if(internalTimer){ 
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }
    switch(approach){
    case 0:
        rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 1:
        rgb_to_grayscale_kernel_int<<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 2:
        rgb_to_grayscale_kernel_float<<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 3:
        rgb_to_grayscale_kernel_float_coalesced<<<cdiv(w/4*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w/4*h);
        }
    if(internalTimer){
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        printf("elTime (ns): %f\n", ms*1000);        
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    return output;
}'''

cpp_src = "torch::Tensor rgb_to_grayscale(torch::Tensor input, int approach, bool internalTimer);"

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)

[o for o in dir(module) if o[0]!='_']

imgc = img.contiguous().cuda()

start = torch.cuda.Event(True)
end = torch.cuda.Event(True)
n_to_warmup = 10
n_to_average = 10
# warmup
for i in range(n_to_warmup):
    res = module.rgb_to_grayscale(imgc, 1, False)


start.record()
for i in range(10):
    res = module.rgb_to_grayscale(imgc, 0, False)
end.record()
end.synchronize()
print(f'Kernel time (double): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

start.record()
for i in range(n_to_average):
    res = module.rgb_to_grayscale(imgc, 1, False)
end.record()
end.synchronize()
print(f'Kernel time (float): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

start.record()
for i in range(n_to_average):
    res = module.rgb_to_grayscale(imgc, 2, False)
end.record()
end.synchronize()
print(f'Kernel time (int): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

start.record()
for i in range(n_to_average):
    res = module.rgb_to_grayscale(imgc, 3, False)
end.record()
end.synchronize()
print(f'Kernel time (coalesced floats): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')