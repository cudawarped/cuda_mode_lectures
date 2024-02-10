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
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

template<typename T>
__device__ unsigned char rgb_to_gray(const unsigned char r, const unsigned char g, const unsigned char b){
    return static_cast<T>(0.2989)*r +  static_cast<T>(0.5870)*g + static_cast<T>(0.1140)*b;
}

template<typename T>
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = rgb_to_gray<T>(x[i], x[i+n], x[i+2*n]);
}

__global__ void rgb_to_grayscale_kernel_int(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = ((77*x[i] + 150*x[i+n] + 29*x[i+2*n]) >> 8);
}

template<typename T>
__device__ void rgb_to_gray(const unsigned int rVec, const unsigned int gVec, const unsigned int bVec, unsigned int* greyVec){
    unsigned char* grey = (unsigned char*)greyVec;
    for (int i = 0; i < 4; i++) {
        const unsigned char r = ((const unsigned char*)&rVec)[i];
        const unsigned char g = ((const unsigned char*)&gVec)[i];
        const unsigned char b = ((const unsigned char*)&bVec)[i];
        grey[i] = rgb_to_gray<T>(r,g,b);
    }
}

// coalesced read/write - example implementation without handler for unaligned memory or width which is not divisible by 4
template<typename T>
__global__ void rgb_to_grayscale_kernel_coalesced(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>n) return;
    unsigned int *xUint = (unsigned int*)x;
    const unsigned int r = xUint[idx];
    const unsigned int g = xUint[idx+n];
    const unsigned int b = xUint[idx+2*n];
    unsigned int gray;
    rgb_to_gray<T>(r,g,b, &gray);
    unsigned int *outUint = (unsigned int*)out;
    outUint[idx] =  gray;
}

// vectorized coalesced read/write, no computation baseline
__global__ void rgb_to_r(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx > n) return;
    uint2 *inUint2 = (uint2*)x;
    uint2 *outUint2 = (uint2*)out;
    outUint2[idx] = inUint2[idx];
}

// vectorized coalesced read/write -  no safety checking
template<typename T>
__global__ void rgb_to_grayscale_kernel_coalesced_uint2(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx > n) return;
    uint2 *xUint = (uint2*)x;
    uint2 greyVec;
    rgb_to_gray<T>(xUint[idx].x, xUint[idx+n].x, xUint[idx+2*n].x, &greyVec.x);
    rgb_to_gray<T>(xUint[idx].y, xUint[idx+n].y, xUint[idx+2*n].y, &greyVec.y);
    uint2 *outUint = (uint2*)out;
    outUint[idx] = greyVec;
}

// cub load/store - no safety checking
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void rgb_to_grayscale_cub(unsigned char* x, unsigned char* out, int n) {
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
    typedef cub::BlockLoad<unsigned char, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE > BlockLoad;
    typedef cub::BlockStore<unsigned char, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> BlockStore;
    __shared__ union TempStorage
    {
        typename BlockLoad::TempStorage    load;
        typename BlockStore::TempStorage   store;
    } temp_storage;

    unsigned char r[ITEMS_PER_THREAD];
    unsigned char g[ITEMS_PER_THREAD];
    unsigned char b[ITEMS_PER_THREAD];
    unsigned char grey[ITEMS_PER_THREAD];
    const int blockOffset = blockIdx.x * TILE_SIZE;
    BlockLoad(temp_storage.load).Load(x + blockOffset, r);
    BlockLoad(temp_storage.load).Load(x + blockOffset + n, g);
    BlockLoad(temp_storage.load).Load(x + blockOffset + 2*n, b);

    for(int i = 0; i < ITEMS_PER_THREAD; i++)
        grey[i] = rgb_to_gray<T>(r[i],g[i],b[i]);

    BlockStore(temp_storage.store).Store(out + blockOffset, grey);
}

void rgb_to_grayscale(torch::Tensor input, torch::Tensor output, int approach, bool internalTimer) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    constexpr int threads = 256;
    cudaEvent_t start, end;
    if(internalTimer){ 
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }
    switch(approach){
    case 0:
        rgb_to_grayscale_kernel<double><<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 1:
        rgb_to_grayscale_kernel<float><<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 2:
        rgb_to_grayscale_kernel_int<<<cdiv(w*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
        break;
    case 3:
        rgb_to_grayscale_kernel_coalesced<float><<<cdiv(w/4*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w/4*h);
            break;
    case 4:
        rgb_to_grayscale_kernel_coalesced_uint2<float><<<cdiv(w/8*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w/8*h);
            break;
    case 5:
        rgb_to_r<<<cdiv(w/8*h,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w/8*h);
            break;
    case 6:
        rgb_to_grayscale_cub<float, threads,8><<<cdiv(w*h/8,threads), threads>>>(
            input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
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
print(f'Kernel time (double for calc): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 1, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (float for calc): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 2, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (int for calc): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 3, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (coalesced uint): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 4, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (vectorized coalesced uint2): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 5, False)
end.record()
end.synchronize()
assert_equal(img[0,:,:], img_grey.cpu())
print(f'Memory Transfer of R plane (vectorized coalesced loads/stores): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')

img_grey.zero_()
start.record()
for i in range(n_to_average):
    module.rgb_to_grayscale(imgc, img_grey, 6, False)
end.record()
end.synchronize()
assert_equal(img_grey_gs, img_grey.cpu())
print(f'Kernel time (vectorized coalesced loads/stores with cub): {start.elapsed_time(end)*1000/n_to_average:.2f}ns')