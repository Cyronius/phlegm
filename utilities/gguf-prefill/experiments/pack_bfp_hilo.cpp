// CPU feasibility probe for fused compensated-BFP input panels. No runtime hook.
#include "open_qwen36/block_host.hpp"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static inline __m256 bf16(__m256 x) {
    auto u=_mm256_castps_si256(x);
    auto odd=_mm256_and_si256(_mm256_srli_epi32(u,16),_mm256_set1_epi32(1));
    u=_mm256_add_epi32(u,_mm256_add_epi32(odd,_mm256_set1_epi32(0x7fff)));
    return _mm256_castsi256_ps(_mm256_and_si256(u,_mm256_set1_epi32(-65536)));
}
// ArtLite's documented host encoding rule, independently implemented with AVX2:
// nearest-even, raise the shared exponent when a positive mantissa rounds to128.
static inline __m256 encode(__m256 x,uint8_t *dst) {
    auto u=_mm256_and_si256(_mm256_castps_si256(x),_mm256_set1_epi32(0x7fffffff));
    auto a=_mm_max_epi32(_mm256_castsi256_si128(u),_mm256_extracti128_si256(u,1));
    a=_mm_max_epi32(a,_mm_srli_si128(a,8));
    a=_mm_max_epi32(a,_mm_srli_si128(a,4));
    unsigned bits=static_cast<unsigned>(_mm_cvtsi128_si32(a));
    if(!bits) { std::memset(dst,0,9);return _mm256_setzero_ps(); }
    unsigned e=std::max(7u,bits>>23);
    auto scale=_mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>((260-e)<<23)));
    auto rounded=_mm256_round_ps(_mm256_mul_ps(x,scale),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    auto q=_mm256_cvttps_epi32(rounded);
    if(_mm256_movemask_epi8(_mm256_cmpgt_epi32(q,_mm256_set1_epi32(127)))) {
        ++e;
        scale=_mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>((260-e)<<23)));
        q=_mm256_cvttps_epi32(_mm256_round_ps(_mm256_mul_ps(x,scale),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
    }
    dst[0]=static_cast<uint8_t>(e);
    auto q16=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst+1),_mm_packs_epi16(q16,q16));
    auto inv=_mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>((e-6)<<23)));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(q),inv);
}
static void packed(const float *x,size_t T,size_t K,uint8_t *out) {
#pragma omp parallel for
    for(long long kb=0;kb<static_cast<long long>(K/64);++kb)
        for(size_t nb=0;nb<T/32;++nb)
            for(size_t nt=0;nt<4;++nt)
                for(size_t ki=0;ki<8;++ki) {
                    auto *p=out+((((kb*(T/32)+nb)*4+nt)*8+ki)*144);
                    for(size_t row=0;row<8;++row) {
                        auto v=bf16(_mm256_loadu_ps(x+(nb*32+nt*8+row)*K+kb*64+ki*8));
                        auto hi=encode(v,p+row*9);
                        encode(bf16(_mm256_sub_ps(v,hi)),p+72+row*9);
                    }
                }
}
template<class F> static double time_ms(F fn) {
    auto start=std::chrono::steady_clock::now();fn();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
int main(int argc,char **argv) {
    if(argc!=6) {std::cerr<<"T K input-f32 output-native output-hilo\n";return 2;}
    size_t T=std::stoul(argv[1]),K=std::stoul(argv[2]);
    if(!T||!K||T%32||K%64) throw std::runtime_error("shape does not tile");
    std::vector<float> x(T*K);
    std::ifstream f(argv[3],std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(x.data()),x.size()*4)) throw std::runtime_error("short input");
    // This probe tests bounded finite values; model integration needs wider edge gates.
    for(auto v:x) if(!(v>=-1e20f&&v<=1e20f)) throw std::runtime_error("input outside probe range");
    std::vector<uint16_t> native(T*K);
    std::vector<uint8_t> hilo(T*K*9/4);
    for(int i=0;i<4;++i) {
        open_qwen36::host::tile_x(x.data(),T,K,native.data());
        packed(x.data(),T,K,hilo.data());
    }
    for(int i=0;i<20;++i) {
        double a,b;
        if(i%2) {b=time_ms([&]{packed(x.data(),T,K,hilo.data());});a=time_ms([&]{open_qwen36::host::tile_x(x.data(),T,K,native.data());});}
        else {a=time_ms([&]{open_qwen36::host::tile_x(x.data(),T,K,native.data());});b=time_ms([&]{packed(x.data(),T,K,hilo.data());});}
        std::cout<<i<<","<<a<<","<<b<<"\n";
    }
    std::ofstream(argv[4],std::ios::binary).write(reinterpret_cast<char*>(native.data()),native.size()*2);
    std::ofstream(argv[5],std::ios::binary).write(reinterpret_cast<char*>(hilo.data()),hilo.size());
}
