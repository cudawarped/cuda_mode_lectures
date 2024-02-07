import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

img = io.read_image('puppy.jpg')
img_grey_gs = (0.2989*img[0,:,:] + 0.5870*img[1,:,:] + 0.1140*img[2,:,:]).type(torch.uint8)

def assert_equal(img_1, img_2, atol = 1):
    assert torch.isclose(img_1,img_2, atol=atol).all()

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

// coalesced read/write using floats for grey calc - example implementation without handler for unaligned memory or width which is not divisible by 4
__global__ void rgb_to_grayscale_kernel_float_coalesced(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>n) return;
    unsigned int *xUint = (unsigned int*)x;
    const unsigned int rVec = xUint[idx];
    const unsigned int gVec = xUint[idx+n];
    const unsigned int bVec = xUint[idx+2*n];
    unsigned int greyVec;
    unsigned char* grey = (unsigned char*)&greyVec;
    for (int i = 0; i < 4; i++) {
        //const unsigned char r = ((rVec >> i * 8) & 0xFFU);
        const unsigned char r = ((const unsigned char*)&rVec)[i];
        const unsigned char g = ((const unsigned char*)&gVec)[i];
        const unsigned char b = ((const unsigned char*)&bVec)[i];
        grey[i] = 0.2989f*r + 0.5870f*g + 0.1140f*b;
    }
    unsigned int *outUint = (unsigned int*)out;
    outUint[idx] = greyVec;
}

// block unrolled and coalesced read/write using floats for grey calc no safety checking
__global__ void rgb_to_grayscale_kernel_float_coalesced_unroll(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int *xUint = (unsigned int*)x;
    const unsigned int rVec = xUint[idx];
    const unsigned int gVec = xUint[idx+n];
    const unsigned int bVec = xUint[idx+2*n];
    unsigned int greyVec;
    unsigned char* grey = (unsigned char*)&greyVec;
    for (int i = 0; i < 4; i++) {
        //const unsigned char r = ((rVec >> i * 8) & 0xFFU);
        const unsigned char r = ((const unsigned char*)&rVec)[i];
        const unsigned char g = ((const unsigned char*)&gVec)[i];
        const unsigned char b = ((const unsigned char*)&bVec)[i];
        grey[i] = 0.2989f*r + 0.5870f*g + 0.1140f*b;
    }
    unsigned int *outUint = (unsigned int*)out;
    outUint[idx] = greyVec;
}

void rgb_to_grayscale(torch::Tensor input, torch::Tensor output, int approach, bool internalTimer) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
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
        printf("elTime (ns): %.2f\n", ms*1000);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
}'''

cpp_src = "void rgb_to_grayscale(torch::Tensor input, torch::Tensor output, int approach, bool internalTimer);"

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)

[o for o in dir(module) if o[0]!='_']

imgc = img.contiguous().cuda()
img_grey = torch.zeros((imgc.size(1),imgc.size(2)), dtype=torch.uint8).cuda()

start = torch.cuda.Event(True)
end = torch.cuda.Event(True)
n_to_warmup = 5
n_to_average = 5
# warmup
for i in range(n_to_warmup):
    module.rgb_to_grayscale(imgc, img_grey, 0, False)

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 0, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (double): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 1, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (float): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 2, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (int): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 3, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (coalesced floats): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')