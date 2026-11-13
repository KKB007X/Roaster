#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "kernels.h"

static uint64_t rs = 88172645463325252ULL;
static uint32_t rnd(void){ rs ^= rs<<13; rs ^= rs>>7; rs ^= rs<<17; return (uint32_t)(rs>>11); }
static float frnd(void){ return ((rnd()&0xFFFFF)/(float)0xFFFFF)*2.f-1.f; }

/* fp16 for small positive values */
static uint16_t h16(float f){ union{float f;uint32_t u;}v={f}; uint32_t e=((v.u>>23)&255)-127+15; return (uint16_t)((e<<10)|((v.u>>13)&0x3FF)); }

static void *alloc_w(size_t n){ size_t g; void *p = kernel_vk_alloc_weights(n,&g); if(!p){ posix_memalign(&p,4096,(n+4095)&~4095UL);} return p; }

static void *make_w(int q6, size_t rows, size_t cols)
{
    size_t bpr=cols/256, bs=q6?210:144, n=rows*bpr*bs;
    uint8_t *w = alloc_w(n);
    for(size_t i=0;i<n;i++) w[i]=(uint8_t)rnd();
    for(size_t b=0;b<rows*bpr;b++){
        uint8_t *blk=w+b*bs;
        if(q6){ uint16_t d=h16(0.002f+0.001f*(rnd()%8)); memcpy(blk+208,&d,2); for(int i=0;i<16;i++) ((int8_t*)blk+192)[i]=(int8_t)((int)(rnd()%64)-32); }
        else { uint16_t d=h16(0.002f+0.001f*(rnd()%8)), m=h16(0.001f*(rnd()%8)); memcpy(blk,&d,2); memcpy(blk+2,&m,2); }
    }
    return w;
}

static double maxrel(const float *a,const float *b,size_t n){ double mx=0,ref=0; for(size_t i=0;i<n;i++){ ref=fmax(ref,fabs(b[i])); mx=fmax(mx,fabs(a[i]-b[i])); } return mx/(ref>0?ref:1); }

static void ref_matmul(int q6,const void*W,const float*X,float*Y,size_t rows,size_t cols,size_t T)
{
    size_t rb=(cols/256)*(q6?210:144);
    for(size_t t=0;t<T;t++) for(size_t r=0;r<rows;r++){
        const uint8_t*row=(const uint8_t*)W+r*rb;
        Y[t*rows+r]= q6? kernel_dot_q6_k(row,X+t*cols,cols): kernel_dot_q4_k(row,X+t*cols,cols);
    }
}

static int fails=0;
static void check(const char*name,double e){ int bad = e>2e-4; printf("  %-34s maxrel=%.2e %s\n",name,e,bad?"FAIL":"ok"); fails+=bad; }

int main(void)
{
    size_t Ts[]={1,3,8,9,16,17,100,257};
    for(int q6=0;q6<2;q6++) for(size_t rows_i=0;rows_i<3;rows_i++){
        size_t rowsv[]={512,2048,37}, cols= rows_i==1?512:2048, rows=rowsv[rows_i];
        void*W=make_w(q6,rows,cols);
        for(size_t ti=0;ti<8;ti++){
            size_t T=Ts[ti];
            float*X=malloc(T*cols*4),*Y=calloc(T*rows,4),*R=malloc(T*rows*4);
            for(size_t i=0;i<T*cols;i++) X[i]=frnd();
            ref_matmul(q6,W,X,R,rows,cols,T);
            KernelMatmulJob j={q6,W,X,Y,rows,cols,T};
            if(kernel_matmul_quant_batch(&j,1)) {puts("batch error");return 2;}
            char nm[96]; snprintf(nm,sizeof nm,"matmul q%d rows=%zu cols=%zu T=%zu",q6?6:4,rows,cols,T);
            check(nm,maxrel(Y,R,T*rows));
            free(X);free(Y);free(R);
        }
    }
    /* several jobs sharing X in one submit */
    {
        size_t T=40,cols=2048; float*X=malloc(T*cols*4); for(size_t i=0;i<T*cols;i++) X[i]=frnd();
        size_t rw[3]={256,64,32}; int q[3]={0,1,0}; void*W[3]; float*Y[3],*R[3]; KernelMatmulJob j[3];
        for(int i=0;i<3;i++){W[i]=make_w(q[i],rw[i],cols);Y[i]=calloc(T*rw[i],4);R[i]=malloc(T*rw[i]*4);ref_matmul(q[i],W[i],X,R[i],rw[i],cols,T);j[i]=(KernelMatmulJob){q[i],W[i],X,Y[i],rw[i],cols,T};}
        kernel_matmul_quant_batch(j,3);
        for(int i=0;i<3;i++){char nm[64];snprintf(nm,sizeof nm,"shared-X job %d",i);check(nm,maxrel(Y[i],R[i],T*rw[i]));}
    }
    /* MoE batch */
    {
        size_t inter=512,hidden=2048; int ne=6; size_t ns[6]={5,1,0,17,9,30}; size_t tot=0; for(int i=0;i<ne;i++) tot+=ns[i];
        KernelMoeExpert ex[6]; float*X=malloc(tot*hidden*4),*D=calloc(tot*hidden,4),*R=calloc(tot*hidden,4);
        for(size_t i=0;i<tot*hidden;i++) X[i]=frnd();
        size_t row=0;
        for(int i=0;i<ne;i++){
            int g6=i%2, d6=(i/2)%2;
            ex[i]=(KernelMoeExpert){make_w(g6,inter,hidden),make_w(0,inter,hidden),make_w(d6,hidden,inter),g6,0,d6,ns[i]};
            if(ns[i]){
                float*g=malloc(ns[i]*inter*4),*u=malloc(ns[i]*inter*4);
                ref_matmul(g6,ex[i].gate,X+row*hidden,g,inter,hidden,ns[i]);
                ref_matmul(0,ex[i].up,X+row*hidden,u,inter,hidden,ns[i]);
                for(size_t k=0;k<ns[i]*inter;k++) g[k]=g[k]/(1+expf(-g[k]))*u[k];
                ref_matmul(d6,ex[i].down,g,R+row*hidden,hidden,inter,ns[i]);
                free(g);free(u);
            }
            row+=ns[i];
        }
        if(kernel_moe_prefill(ex,ne,X,D,inter,hidden)) {puts("moe error");return 2;}
        check("moe prefill 6 experts",maxrel(D,R,tot*hidden));
    }
    printf(fails?"FAILED (%d)\n":"ALL OK\n",fails);
    kernel_vk_shutdown();
    return fails!=0;
}
