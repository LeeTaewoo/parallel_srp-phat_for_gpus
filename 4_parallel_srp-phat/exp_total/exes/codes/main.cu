// Release date: June 2015
// Author: Taewoo Lee, (twlee@speech.korea.ac.kr)
//
// Copyright (C) 2015 Taewoo Lee
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
// Reference:
// [1] Taewoo Lee, Sukmoon Chang, and Dongsuk Yook, "Parallel SRP-PHAT for
//     GPUs," Computer Speech and Language, vol. 35, pp. 1-13, Jan. 2016.
//
#include "main.h"

int Answer;
int ErrorTolerance= 10;
const char* deviceFileName="./exp_total/getDeviceInfo/deviceInfo.bin";
const char* datFileList="./datFileListToProcess.txt";
texture<float> texDevCC;

char4 h_tdoa_table_char4[POWER_SIZE*(N/4)];
int h_TDOA_table[Q*N];
float h_TOA_table[Q*M];
float h_SRP[Q];
int32_t h_sphCoords[Q*3];


int32_t main(int32_t argc, char **argv) {
	if (argc!=3) {
		fprintf(stderr,"\nUsage: exp [direction(deg)] [error_torlerance(deg)]\n\n"); 
		exit(-1);
	}
	Answer= strtol(argv[1],0,10);
	ErrorTolerance= strtol(argv[2],0,10);
	printf("Answer=%ld, ErrorTolerance=%ld, ", Answer,ErrorTolerance);
  load_sphCoords();
#if TD_GPU==1
  printf("TD\n");
	load_TDOA_table();
  srp_phat_TDGPU();
#endif
#if FD_GPU==1
  printf("FD\n");
	load_TOA_table();
	srp_phat_FDGPU();
#endif
	return 0;
}


void srp_phat_FDGPU(void) {
  FILE *fp;
  int32_t n;
  cudaDeviceProp deviceProp;
  size_t gpuMemSize=0;
  cudaError_t err;
  cufftHandle plan;
  cufftPlan1d(&plan,T,CUFFT_C2C,M);
  
  fp= fopen(deviceFileName,"rb"); assert(fp!=0);
  n= fread(&deviceProp,sizeof(cudaDeviceProp),1,fp); assert(n==1); n=0;
  fclose(fp);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(fdsp_srp_gpu4,cudaFuncCachePreferShared);

  int32_t tmp1=(T*M)/128, tmp2=128;
  dim3 dimGridWin(tmp1,1,1), dimBlockWin(tmp2,1,1); //for applyWin
#if __CUDA_ARCH__ < 300
  fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,"CUDA Compute Capability must >= 3.0");
#endif
  dim3 dimFDSPMi(Q,1,1);              //for FD_SRP_Minotto
  int dim_blk= (int)(Q/1024.0f+0.5f); // for FD_Proposed

  char dat_file_name[100];
  fp= fopen(datFileList,"r"); assert(fp!=0);
  n= fscanf(fp,"%s",&dat_file_name); assert(n!=0); n=0;
  printf("%s\n",dat_file_name);
  FILE *fp2= fopen(dat_file_name,"rb"); assert(fp2!=0);
  int32_t nFrames, t;
  n= fread(&nFrames,sizeof(int32_t),1,fp2); assert(n==1); n=0;
  n= fread(&t,sizeof(int32_t),1,fp2); assert(n==1); n=0;
  fclose(fp2);  
  int16_t* dataBlock=(int16_t*)malloc(sizeof(int16_t)*M*nFrames*T); assert(dataBlock!=0);
  for (int32_t i=0; i<M; ++i) {
    n= fscanf(fp,"%s",&dat_file_name); assert(n!=0); n=0;
    printf("%s\n",dat_file_name);
    fp2= fopen(dat_file_name,"rb"); assert(fp2!=0);
    n= fread(dataBlock+(i*nFrames*T),sizeof(int16_t),nFrames*T,fp2); assert(n==nFrames*T); n=0;
    fclose(fp2);
  }
  printf("dataBlock size: (M=%d) x (nFrames=%d) x (T=%d).\n",M,nFrames,T);
  fclose(fp);

  //////////////////////////////////////////////////////////////////////////////////
  /////////////////////// GPU MEM INIT START ///////////////////////////////////////
  cufftComplex *h_dataBlock= (cufftComplex*)malloc(sizeof(cufftComplex)*M*T);
  cufftComplex *d_dataBlock;
  err= cudaMalloc((void**)&d_dataBlock, sizeof(cufftComplex)*M*T);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float h_window[M*T];
  for (int32_t i=0; i<M; ++i) {
    for (int32_t j=0; j<T; ++j) {	
      float d= 0.54f - 0.46f*cos(2.0f*(float)M_PI*(float)j/(float)T);
      h_window[(i*T)+j]= d;
    }
  }
  float *d_window;
  err= cudaMalloc((void**)&d_window, sizeof(float)*M*T);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemcpy(d_window, h_window, sizeof(float)*M*T,cudaMemcpyHostToDevice);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float *d_power;
  err= cudaMalloc((void**)&d_power, sizeof(float)*POWER_SIZE);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float h_w[T]; //omega
  int32_t cnt=0;
  for (float i=0.0f; i<=2.0f*(float)M_PI; i+=(2.0f*(float)M_PI)/((float)T)) {
	  h_w[cnt++]= i;
  }
  float *d_w;
  err= cudaMalloc((void**)&d_w, sizeof(float)*T);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemcpy(d_w, h_w, sizeof(float)*T, cudaMemcpyHostToDevice);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float *d_TOA_table;
  err= cudaMalloc((void**)&d_TOA_table, sizeof(float)*Q*M);
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemcpy(d_TOA_table,h_TOA_table,sizeof(float)*Q*M,cudaMemcpyHostToDevice);
  if (cudaGetLastError()!=cudaSuccess) {
  fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  int* d_max_indices;
  err= cudaMalloc((void**)&d_max_indices, sizeof(int)*(POWER_SIZE/(NW*4)));
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemset(d_max_indices, 0, sizeof(int)*(POWER_SIZE/(NW*4)));
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float* d_max_powers;
  err= cudaMalloc((void**)&d_max_powers, sizeof(float)*(POWER_SIZE/(NW*4)));
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemset(d_max_powers, 0, sizeof(float)*(POWER_SIZE/(NW*4)));
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  int* d_maxIdx;
  err= cudaMalloc((void**)&d_maxIdx, sizeof(int)*1);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemset(d_maxIdx, 0, sizeof(int)*1);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float *d_power_tmp;
  err= cudaMalloc((void**)&d_power_tmp, sizeof(float)*POWER_SIZE*32); //32=FDGPU4 blk size
  if (cudaGetLastError()!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  /////////////////////// GPU MEM INIT END /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////


  struct timeval start_t;
  gettimeofday(&start_t,NULL);

  int32_t nCorrectFrame=0;
  int h_maxIdx[1];
  for (int32_t n=0; n<nFrames; ++n) {

    for (int32_t ci=0; ci<M; ++ci) {
      for (int32_t i=0; i<T; ++i) {
        h_dataBlock[(ci*T)+i].x= (float)(*(dataBlock+(ci*nFrames*T)+(n*T)+(i)));
        h_dataBlock[(ci*T)+i].y= 0.0f;
      }
    }

	  cudaMemcpy(d_dataBlock,h_dataBlock,sizeof(cufftComplex)*M*T,cudaMemcpyHostToDevice);
	  applyWindow<<<M*32,128>>>(d_dataBlock,d_window);
	  cufftExecC2C(plan,d_dataBlock,d_dataBlock,CUFFT_FORWARD);
	  phat_weight<<<M*32,128>>>(d_dataBlock);

#if FD_GPU_SRP==2   // FD_Minotto
    err= cudaMemset(d_power,0,POWER_SIZE);
		if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
	  fdsp_srp_minotto<<<dimFDSPMi,64>>>(d_dataBlock,d_w,d_TOA_table,d_power);
	  if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#elif FD_GPU_SRP==4   // FD_Proposed
	  fdsp_srp_gpu4<<<32,128>>>(d_dataBlock,d_w,d_TOA_table,d_power_tmp);
	  if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    fdsp_srp_gpu4_sum<<<POWER_SIZE/128,128>>>(d_power_tmp,d_power);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }    
#endif
		err= cudaMemset(d_power+Q-1,0,sizeof(float)*(POWER_SIZE-Q));
    if (err!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

		///////////////////// Find maximum coordinate ///////////////////////////
#if Q<=97200		
		find_max_by_reduction<<<POWER_SIZE/128,128>>>(d_power,d_max_indices,d_max_powers);
		if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		find_max_by_reduction32<<<1,POWER_SIZE/128>>>(d_max_indices,d_max_powers,d_maxIdx);
    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#elif Q==388800
    find_max_by_reduction<<<POWER_SIZE/1024,1024>>>(d_power,d_max_indices,d_max_powers);
		if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		find_max_by_reduction32<<<1,POWER_SIZE/1024>>>(d_max_indices,d_max_powers,d_maxIdx);
    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif    		
    err= cudaMemcpy(h_maxIdx,d_maxIdx,sizeof(int),cudaMemcpyDeviceToHost);
		if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

		int32_t r= h_sphCoords[(h_maxIdx[0]*3)+0];
		int32_t theta= h_sphCoords[(h_maxIdx[0]*3)+1];
		int32_t phi= h_sphCoords[(h_maxIdx[0]*3)+2];
    printf("t= %d, p= %d, r=%d ,%d \n", theta,phi,r,h_maxIdx[0]+1);
		if ((abs(theta-Answer)<=ErrorTolerance)) ++nCorrectFrame;
  }

	struct timeval end_t;
	gettimeofday(&end_t,0);
	double elapsedTime;
	elapsedTime= (double)(end_t.tv_sec)+(double)(end_t.tv_usec)/1000000.0 - \
	             (double)(start_t.tv_sec)-(double)(start_t.tv_usec)/1000000.0;
	
	struct timeval start_t2;
  gettimeofday(&start_t2,NULL);
  for (int32_t n=0; n<nFrames; ++n) {
    for (int32_t ci=0; ci<M; ++ci) {
      for (int32_t i=0; i<T; ++i) {
        h_dataBlock[(ci*T)+i].x= (float)(*(dataBlock+(ci*nFrames*T)+(n*T)+(i)));
        h_dataBlock[(ci*T)+i].y= 0.0f;
      }
    }
  }
  struct timeval end_t2;
	gettimeofday(&end_t2,0);

	double elapsedTime2;
	elapsedTime2= (double)(end_t2.tv_sec)+(double)(end_t2.tv_usec)/1000000.0 - \
	              (double)(start_t2.tv_sec)-(double)(start_t2.tv_usec)/1000000.0;
	
	printf("Execution time : %f (sec)\n", elapsedTime-elapsedTime2);
	printf("Accuracy= %d/%ld, (%.2lf%%)\n",nCorrectFrame,nFrames, 
	                        (double)nCorrectFrame/(double)nFrames*100.0);
	printf("\n\n");

	if (dataBlock) free(dataBlock);
	if (h_dataBlock) free(h_dataBlock);
	if (d_power) cudaFree(d_power);
  if (d_power_tmp) cudaFree(d_power_tmp);
  if (d_window) cudaFree(d_window);
  if (d_dataBlock) cudaFree(d_dataBlock);
  if (d_w) cudaFree(d_w);
  if (d_TOA_table) cudaFree(d_TOA_table);
  if (d_max_indices) cudaFree(d_max_indices);
  if (d_max_powers) cudaFree(d_max_powers);
  if (d_maxIdx) cudaFree(d_maxIdx);
  if (plan) cufftDestroy(plan);
	cudaDeviceReset();
}



__global__ void phat_weight(cufftComplex* d_dataBlock) {
  int bx=blockIdx.x, tx=threadIdx.x, bdx=blockDim.x;
  cufftComplex tmp= d_dataBlock[ bx*bdx + tx];
  float mag= ComplexAbs(tmp);
  tmp.x /= mag;
  tmp.y /= mag;
  d_dataBlock[(bx*bdx) + (tx)]= tmp;
}

__global__ void fdsp_srp_minotto(const cufftComplex* d_dataBlock,
                                 const float* d_w, 
                                 const float* d_TOA_table, 
                                 float* d_power) {
	int tx= threadIdx.x;
	int q= blockIdx.x;
	__shared__ float sum[64];
	
	float localSum= 0.0f;	
	for (int i=0; i<64; ++i) {
    float2 tmpSum= {0.0f,0.0f};
    for (int ci=0; ci<M; ++ci) {	
      cufftComplex tmp= d_dataBlock[(ci*T)+(i*64)+tx];
      float wt= d_w[(i*64)+tx] * d_TOA_table[(q*M)+ci];
      float a= tmp.x;
      float b= tmp.y;
      float c= cos(wt);
      float d= sin(wt);
      tmpSum.x += (a*c)-(b*d);
      tmpSum.y += (a*d)+(b*c);
    }
    localSum += (tmpSum.x*tmpSum.x) + (tmpSum.y*tmpSum.y);
  }	
  sum[tx]= localSum;
  __syncthreads();

  int i= 64>>1;
  while (i!=0) {
    if (tx<i) { 
      sum[tx]+= sum[tx+i];
    }
    __syncthreads();
    i= i>>1;
  }	
  if (tx==0) { 
    d_power[q]= sum[0];
  }
}

// <<<32,128>>>, 48KB shMEM, (T/128)*128=4096
__global__ void fdsp_srp_gpu4(const cufftComplex* d_dataBlock,
                              const float* d_w, 
                              const float* d_TOA_table, 
                              float* d_power_tmp)	{
  int bx=blockIdx.x, tx=threadIdx.x;
#if ((1*SUBBAND)*SIZE_OF_FLOAT)+((SUBBAND*M)*SIZE_OF_FLOAT2) <= SHMEM_SIZE_MAX
  __shared__ float s_w[SUBBAND];            //128*4B=512B
  __shared__ cufftComplex s_X[SUBBAND*M];   //M*128*8B=32*128*8B=32KB
#endif
  float tauTable[M];                        //M*4B=32*4B=128B

  s_w[tx]= d_w[(bx*SUBBAND)+tx];            //data load onto each shared memory
  for (int m=0; m<M; ++m) {
    s_X[(tx*M)+m]= d_dataBlock[(m*T)+(bx*SUBBAND)+(tx)];
  }
  __syncthreads();

#if Q==3888                 //#Q per thread 
  int nq=30;                //Q=3888, 128*30=3840(+48)=3888
#elif Q==97200          
  int nq=758;               //Q=97200, 128*759=97152(+48)=97200
#elif Q==388800       
  int nq=3037;              //Q=388800, 128*3037=388736(+64)=388800
#endif
             
  for (int i=0; i<nq; ++i) {
    int q= (tx*nq)+i;
    for (int m=0; m<M; ++m) {             //load onto registers
      tauTable[m]= d_TOA_table[(q*M)+m];
    }
    float subsrp= 0.0f;
    for (int k=0; k<SUBBAND; ++k) {
      float omega= s_w[k];
      Complex sum= {0.0f,0.0f};
      for (int m=0; m<M; ++m) {
        float wt= omega * tauTable[m];
        float2 tmp2= s_X[(k*M)+(m)];
        float a= tmp2.x;
        float b= tmp2.y;
        float c= cos(wt);
        float d= sin(wt);
        sum.x += (a*c)-(b*d);
        sum.y += (a*d)+(b*c);
      }
      subsrp += (sum.x*sum.x) + (sum.y*sum.y);
    }
    d_power_tmp[(q*32)+bx]= subsrp;
  }

#if Q==3888                               //Processing remained coordinates
  if (tx>=48) return;
  nq= 3840; 
#elif Q==97200
  if (tx>=48) return;
  nq= 97152;
#elif Q==388800
  if (tx>=64) return;
  nq= 388736;
#endif
  int q= nq+tx;
  for (int m=0; m<M; ++m) {
   tauTable[m]= d_TOA_table[(q*M)+m];
  }
  float subsrp= 0.0f;
    for (int k=0; k<SUBBAND; ++k) {
      float omega= s_w[k];
      Complex sum= {0.0f,0.0f};
      for (int m=0; m<M; ++m) {
        float wt= omega * tauTable[m];
        float2 tmp2= s_X[(k*M)+(m)];
        float a= tmp2.x;
        float b= tmp2.y;
        float c= cos(wt);
        float d= sin(wt);
        sum.x += (a*c)-(b*d);
        sum.y += (a*d)+(b*c);
      }
      subsrp += (sum.x*sum.x) + (sum.y*sum.y);
    }
    d_power_tmp[(q*32)+bx]= subsrp;
}


//<<<POWER_SIZE/128,128>>>
//Q==3888->4096, 32blks * 128threads= 4096threads
//Q==97200->131072, 1024blks * 128threads= 131072threads
//Q==388800->524288, 128blks * 4096threads= 524288threads
//                   gdx       bdx
__global__ void fdsp_srp_gpu4_sum(const float* d_power_tmp,
                                  float* d_power) {
  int bx=blockIdx.x, tx=threadIdx.x, bdx= blockDim.x;
  int q= (bx*bdx)+(tx);
  float sum= 0.0f;  
  for (int i=0; i<32; ++i) {        //32==FDGPU4 #blks
    sum += d_power_tmp[(q*32)+i];
  }
  d_power[q]= sum;
}


/////////////////////////////////////////////////////////////////////////////
//////////////////////////////   TD GPU   ///////////////////////////////////

void srp_phat_TDGPU(void) {
  FILE *fp;
  int32_t n;
  cudaDeviceProp deviceProp;
  size_t gpuMemSize=0;
  cudaError_t err;
  cufftResult errCufft;

  fp= fopen(deviceFileName,"rb"); assert(fp!=0);
  n= fread(&deviceProp,sizeof(cudaDeviceProp),1,fp); assert(n==1); n=0;
  fclose(fp);
  err= cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
#if (CC_PAIR_SIZE*N*SIZE_OF_FLOAT) > 16384
  err= cudaFuncSetCacheConfig(SRP_TD_GPU4_3, cudaFuncCachePreferShared);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
#endif
#if (NEXT_POW_OF_2(N)*SIZE_OF_FLOAT) > 16384
  err= cudaFuncSetCacheConfig(SRP_Minotto, cudaFuncCachePreferShared);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
#endif

  int tmp1= (T*M)/(NW*(int)4); 
  int tmp2= NW*(int)4; 
  dim3 dimGridWin(tmp1,1,1);			  // for applyWin
  dim3 dimBlockWin(tmp2,1,1);		    // for applyWin
  dim3 dimGridCC(1,N,1);		        // for copyDevCC 
  dim3 dimBlockCC(MAX_SD,1,1);			// for copyDevCC
#if Q==388800
  assert(Q < 388800+1);
  dim3 dimSRPMinottoBlk(6,65535,1);
#elif Q==97200  
  dim3 dimSRPMinottoBlk(2,65535,1);
#elif Q<=32400
  dim3 dimSRPMinottoBlk(1,Q,1);	
#endif
#if N>1024
  dim3 dimSRPMinottoThr(1024,1,1);
#else
  dim3 dimSRPMinottoThr(N,1,1);
#endif  

  char dat_file_name[100];
  fp= fopen(datFileList,"r"); assert(fp!=0);
  n= fscanf(fp,"%s",&dat_file_name); assert(n!=0); n=0;
  printf("%s\n",dat_file_name);
  FILE *fp2= fopen(dat_file_name,"rb"); assert(fp2!=0);
  int32_t nFrames, t;
  n= fread(&nFrames,sizeof(int32_t),1,fp2); assert(n==1); n=0;
  n= fread(&t,sizeof(int32_t),1,fp2); assert(n==1); n=0;
  fclose(fp2);  
  int16_t* dataBlock=(int16_t*)malloc(sizeof(int16_t)*M*nFrames*T); assert(dataBlock!=0);
  for (int32_t i=0; i<M; ++i) {
    n= fscanf(fp,"%s",&dat_file_name); assert(n!=0); n=0;
    printf("%s\n",dat_file_name);
    fp2= fopen(dat_file_name,"rb"); assert(fp2!=0);
    n= fread(dataBlock+(i*nFrames*T),sizeof(int16_t),nFrames*T,fp2); assert(n==nFrames*T); n=0;
    fclose(fp2);
  }
  printf("dataBlock size: (M=%d) x (nFrames=%d) x (T=%d).\n",M,nFrames,T);
  fclose(fp);

  //////////////////////////////////////////////////////////////////////////////////
  /////////////////////// GPU MEM INIT START ///////////////////////////////////////
	cufftComplex *h_dataBlock= (cufftComplex*)malloc(sizeof(cufftComplex)*M*T);
	cufftComplex *d_dataBlock=0;
	err= cudaMalloc((void**)&d_dataBlock,sizeof(cufftComplex)*M*T);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float h_window[T*M];
  for(int32_t i=0; i<M; ++i) {
    for(int32_t j=0; j<T; ++j) {	
		  h_window[i*T + j]= (0.54f-0.46f*cos(2.0f*M_PI*j/(float)T)); 
	  }
  }
	float *d_window=0;
	err= cudaMalloc((void**)&d_window, sizeof(float)*T*M);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemcpy(d_window, h_window, sizeof(float)*T*M,cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	cufftComplex *d_ifft=0;
	err= cudaMalloc((void**)&d_ifft,sizeof(cufftComplex)*N*T);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }	                      
	err= cudaMemset(d_ifft,0,sizeof(cufftComplex)*T*N);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	int h_pair[2*N];
	int32_t cnt=0;
	for (int32_t i=0; i<M-1; ++i) {
		for (int32_t j=i+1; j<M; ++j) {
			h_pair[cnt*2+0]= i;
			h_pair[cnt*2+1]= j;
			++cnt;
		}
	}
	int *d_pair=0;
	err= cudaMalloc((void**)&d_pair,sizeof(int)*2*N);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemset(d_pair,0,sizeof(int)*2*N);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemcpy(d_pair,h_pair,sizeof(int)*2*N,cudaMemcpyHostToDevice);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	cufftHandle plan;
	errCufft= cufftPlan1d(&plan,T,CUFFT_C2C,M);
	if (errCufft!=CUFFT_SUCCESS) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	cufftHandle plan2;
	errCufft= cufftPlan1d(&plan2,T,CUFFT_C2C,N);
	if (errCufft!=CUFFT_SUCCESS) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	cufftHandle plan4;
	errCufft= cufftPlan1d(&plan4,T,CUFFT_C2C,1);
	if (errCufft!=CUFFT_SUCCESS) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	float *d_cc=0;
	err= cudaMalloc((void**)&d_cc,sizeof(float)*CC_PAIR_SIZE*N);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemset(d_cc,0,sizeof(float)*CC_PAIR_SIZE*N);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	float *d_power=0;
	err= cudaMalloc((void**)&d_power,sizeof(float)*POWER_SIZE);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }	
	err= cudaMemset(d_power,0,sizeof(float)*POWER_SIZE);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaBindTexture(NULL,texDevCC,d_cc,sizeof(float)*CC_PAIR_SIZE*N);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	int *d_TDOA_table=0;
	err= cudaMalloc((void**)&d_TDOA_table,sizeof(int)*(POWER_SIZE*N));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemset(d_TDOA_table,0,sizeof(int)*(POWER_SIZE*N));
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	err= cudaMemcpy(d_TDOA_table,h_TDOA_table,sizeof(int)*(Q*N),cudaMemcpyHostToDevice);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
	char4 *d_tdoa_table_char4=0;
	err= cudaMalloc((void**)&d_tdoa_table_char4,sizeof(char4)*(POWER_SIZE*N/4));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }	            
	err= cudaMemset(d_tdoa_table_char4,0,sizeof(char4)*(POWER_SIZE*N/4));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }	                
	err= cudaMemcpy(d_tdoa_table_char4,h_tdoa_table_char4,sizeof(char4)*(Q*N/4),cudaMemcpyHostToDevice);
	if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  int* d_max_indices=0;
  err= cudaMalloc((void**)&d_max_indices,sizeof(int)*(POWER_SIZE/(NW*4)));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }                        
  err= cudaMemset(d_max_indices,0,sizeof(int)*(POWER_SIZE/(NW*4)));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  float* d_max_powers=0;
  err= cudaMalloc((void**)&d_max_powers,sizeof(float)*(POWER_SIZE/(NW*4)));
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }                        
  err= cudaMemset(d_max_powers,0,sizeof(float)*(POWER_SIZE/(NW*4)));
  if (err != cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  int* d_maxIdx=0;
  err= cudaMalloc((void**)&d_maxIdx, sizeof(int)*1);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err= cudaMemset(d_maxIdx,0,sizeof(int)*1);
  if (err!=cudaSuccess) {
    fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  /////////////////////// GPU MEM INIT END /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////

  struct timeval start_t;
  gettimeofday(&start_t,NULL);

  int32_t nCorrectFrame= 0;
  int h_maxIdx[1];
  for (int32_t n=0; n<nFrames; ++n) {
    
    for (int32_t ci=0; ci<M; ++ci) {
      for (int32_t i=0; i<T; ++i) {
        h_dataBlock[(ci*T)+i].x= (float)(*(dataBlock+(ci*nFrames*T)+(n*T)+(i)));
        h_dataBlock[(ci*T)+i].y= 0.0f;
      }
    }

    err= cudaMemcpy(d_dataBlock,h_dataBlock,sizeof(cufftComplex)*M*T,cudaMemcpyHostToDevice);
    if (err!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		applyWindow<<<dimGridWin,dimBlockWin>>>(d_dataBlock,d_window);
		if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		errCufft= cufftExecC2C(plan,d_dataBlock,d_dataBlock,CUFFT_FORWARD);
		if (errCufft!=CUFFT_SUCCESS) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    phat_weight<<<M*32,128>>>(d_dataBlock);
    if (errCufft!=CUFFT_SUCCESS) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    /////////////////////     CROSS CORRELATION     //////////////////////////
#if TD_GPU_CC==13
#if (TD_GPU_CC==13) && (N>65535)
    assert(N<=65535);
#endif
    ccMul_Minotto<<<N,64>>>(d_dataBlock,d_ifft,d_pair);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    errCufft= cufftExecC2C(plan2,d_ifft,d_ifft,CUFFT_INVERSE);
    if (errCufft!=CUFFT_SUCCESS) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#elif TD_GPU_CC==3
#if (TD_GPU_CC==3) && ((256*M*SIZE_OF_FLOAT2) > REGISTER_SIZE_MAX)
    fprintf(stderr,"Warning: register spilling occured at %s %d\n",__FILE__,__LINE__+2);
#endif
    ccMul_shmem_reg<<<16,256>>>(d_dataBlock,d_ifft);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    errCufft= cufftExecC2C(plan2,d_ifft,d_ifft,CUFFT_INVERSE);
    if (errCufft!=CUFFT_SUCCESS) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
    copyDevCC<<<dimGridCC,dimBlockCC>>>(d_ifft,d_cc);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    /////////////////////////     SRP     ///////////////////////////////////
#if TD_GPU_SRP==41  //valid M<=45
#if (NEXT_POW_OF_2(N)*SIZE_OF_FLOAT <= SHMEM_SIZE_MAX)
    SRP_Minotto<<<dimSRPMinottoBlk,dimSRPMinottoThr>>>(d_TDOA_table,d_power);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
#elif TD_GPU_SRP==43
    SRP_TD_GPU4_3<<<POWER_SIZE/1024,1024>>>(d_cc,d_tdoa_table_char4,d_power);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
    err= cudaMemset(d_power+Q-1,0,sizeof(float)*(POWER_SIZE-Q));
    if (err!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

		/////////////////    FIND MAXIMUM SRP COORDINATES    /////////////////////
#if Q<=97200		
		find_max_by_reduction<<<POWER_SIZE/(NW*4),NW*4>>>(d_power,d_max_indices,d_max_powers);
		if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		find_max_by_reduction32<<<1,POWER_SIZE/(NW*4)>>>(d_max_indices,d_max_powers,d_maxIdx);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#elif Q==388800
    find_max_by_reduction<<<POWER_SIZE/1024,1024>>>(d_power,d_max_indices,d_max_powers);
		if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
		find_max_by_reduction32<<<1,POWER_SIZE/1024>>>(d_max_indices,d_max_powers,d_maxIdx);
    if (cudaGetLastError()!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif    
		err= cudaMemcpy(h_maxIdx, d_maxIdx, sizeof(int),cudaMemcpyDeviceToHost);
    if (err!=cudaSuccess) {
      fprintf(stderr,"%s %d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }			                                         

		int32_t r= h_sphCoords[h_maxIdx[0]*3 + 0];
		int32_t t= h_sphCoords[h_maxIdx[0]*3 + 1];
		int32_t p= h_sphCoords[h_maxIdx[0]*3 + 2];
		printf("t= %d, p= %d, r= %d\n", t,p,r);
		if (abs(t-Answer)< ErrorTolerance) {++nCorrectFrame;}
  }


  struct timeval end_t;
  gettimeofday(&end_t,NULL);
  double elapsedTime;	
  elapsedTime= (double)(end_t.tv_sec)+(double)(end_t.tv_usec)/1000000.0 - \
               (double)(start_t.tv_sec)-(double)(start_t.tv_usec)/1000000.0;
               
  struct timeval start_t2;
  gettimeofday(&start_t2,NULL);
  for (int32_t n=0; n<nFrames; ++n) {
    for (int32_t ci=0; ci<M; ++ci) {
      for (int32_t i=0; i<T; ++i) {
        h_dataBlock[(ci*T)+i].x= 
                (float)(*(dataBlock+(ci*nFrames*T)+(n*T)+(i)));
        h_dataBlock[(ci*T)+i].y= 0.0f;
      }
    }
  }
  struct timeval end_t2;
  gettimeofday(&end_t2,NULL);
  double elapsedTime2;	
  elapsedTime2= (double)(end_t2.tv_sec)+(double)(end_t2.tv_usec)/1000000.0 - \
               (double)(start_t2.tv_sec)-(double)(start_t2.tv_usec)/1000000.0;
                              
  printf("Execution time : %f (sec)\n", elapsedTime-elapsedTime2);
  printf("Accuracy= %d/%ld, (%.2lf%%)\n", nCorrectFrame, nFrames,(double)nCorrectFrame/(double)nFrames*100.0);
  printf("\n");

  if (d_max_indices) cudaFree(d_max_indices);
  if (d_max_powers) cudaFree(d_max_powers);
  if (d_maxIdx) cudaFree(d_maxIdx);
  if (dataBlock) free(dataBlock);
  if (h_dataBlock) free(h_dataBlock);
  if (d_dataBlock) cudaFree(d_dataBlock);
  if (d_window) cudaFree(d_window);
  if (d_ifft) cudaFree(d_ifft);
  if (d_pair) cudaFree(d_pair);
  if (plan2) cufftDestroy(plan2);
  if (d_cc) cudaFree(d_cc);
  if (d_power) cudaFree(d_power);
  if (d_TDOA_table) cudaFree(d_TDOA_table);
  if (d_tdoa_table_char4) cudaFree(d_tdoa_table_char4);
  cudaDeviceReset();
}



__global__ void applyWindow(cufftComplex* d_dataBlock, const float* d_window) {
  int idx= blockIdx.x*blockDim.x + threadIdx.x;
	d_dataBlock[idx].x *= d_window[idx];
}

// <<<N,64>>>
__global__ void ccMul_Minotto(const cufftComplex* d_dataBlock, 
                              cufftComplex* d_ifft, 
                              const int* d_pair) {
  int bx=blockIdx.x, tx=threadIdx.x;
  int p1= d_pair[bx*2 + 0];
  int p2= d_pair[bx*2 + 1];
  for (int i=0; i<64; ++i) {
    cufftComplex tmp= ComplexMul(d_dataBlock[(p1*T)+ (tx*64)+ i],
                      ComplexConj(d_dataBlock[(p2*T)+ (tx*64) +i]));
    d_ifft[(bx*T) + (tx*64)+ i]= tmp;
  }
}

__global__ void ccMul_shmem_reg(const cufftComplex* d_dataBlock, 
                                cufftComplex* d_ifft)	{
  cufftComplex s_dat[M];
  for (int i=0; i<M; ++i) {
    s_dat[i]= d_dataBlock[(i*T)+(blockIdx.x*blockDim.x)+(threadIdx.x)];
  }
  int pi=0;
  for (int m1=0; m1<M-1; ++m1) {
    cufftComplex tmp1= s_dat[m1];
    for (int m2=m1+1; m2<M; ++m2) {
      cufftComplex tmp2= ComplexConj(s_dat[m2]);
      cufftComplex tmp= ComplexMul(tmp1, tmp2);
      d_ifft[(pi*T)+(blockIdx.x*blockDim.x)+threadIdx.x]= tmp;
      ++pi;
    }
  }
}

__global__ void copyDevCC(cufftComplex* d_ifft,float* d_cc) {
  int by=blockIdx.y, tx=threadIdx.x;
  d_cc[(by*CC_PAIR_SIZE)+tx]= d_ifft[(by*T)+(T-MAX_SD)+tx].x;
  d_cc[(by*CC_PAIR_SIZE)+(MAX_SD+1)+tx]= d_ifft[ (by*T) +1+tx].x;
  if (tx==0) {
    d_cc[(by*CC_PAIR_SIZE)+MAX_SD]= d_ifft[(by*T)+0].x;		
  }
}

// GPU_SRP==41
__global__ void SRP_Minotto(const int* d_TDOA_table, float* d_power) {
  int bx=blockIdx.x, by=blockIdx.y, tx=threadIdx.x, gdy=gridDim.y;
  __shared__ float srp[NEXT_POW_OF_2(N)];
#if N<=1024
  int sd= d_TDOA_table[(bx*gdy*N)+(by*N)+(tx)];
  srp[tx]= tex1Dfetch(texDevCC, CC_PAIR_SIZE*tx + sd);
  if (tx==0) {
    for (int i=0; i<NEXT_POW_OF_2(N)-N; ++i) {
      srp[N + i]= 0.0f;
    }
  }
  __syncthreads();
  int i= (int)(NEXT_POW_OF_2(N)) >> 1;
  while (i!=0) {
    if (tx<i) {
      srp[tx] += srp[tx+i];
    }
    __syncthreads();
    i= i>>1;
  }
  if (tx==0) { 
    d_power[bx*gdy + by]= srp[0];
  }
#else
  int iter= NEXT_POW_OF_2(N)/2;
  for (int i=0; i<iter; ++i) {
    if ((iter*tx+i) < N) {
      int sd= d_TDOA_table[(bx*gdy*N)+(by*N)+((iter*tx)+i)];
      srp[(iter*tx)+i]= tex1Dfetch(texDevCC,CC_PAIR_SIZE*(iter*tx+i)+sd);
      if (tx==0) {
        for (int i=0; i<NEXT_POW_OF_2(N)-N; ++i) {
          srp[N+i]= 0.0f;
        }
      }
    }
  }
  __syncthreads();
  if (tx==0) {
    for (int i=1; i<N; ++i) {
      srp[0] += srp[i];
    }
    d_power[bx*gdy + by]= srp[0];
  }
#endif  
}

__global__ void SRP_TD_GPU4_3(const float* d_cc, 
                              const char4* d_tdoa_table_char4, 
                              float* d_power) {
  int bx=blockIdx.x, bdx=blockDim.x, tx=threadIdx.x;
#if (CC_PAIR_SIZE*N*SIZE_OF_FLOAT) <= SHMEM_SIZE_MAX 
  __shared__ float cc[CC_PAIR_SIZE*N];	//33*120*4B=15KB
  if (tx<N) {
	  for (int i=0; i<CC_PAIR_SIZE; ++i) {
		  cc[tx*CC_PAIR_SIZE + i]= d_cc[tx*CC_PAIR_SIZE + i];
	  }
  }
  __syncthreads();
#endif
  int q= (bx*bdx) + (tx);
  float srp=0.0f;
  for (int i=0; i<N/4; ++i) {
    char4 tmp= d_tdoa_table_char4[(q*N/4) + i];
#if (CC_PAIR_SIZE*N*SIZE_OF_FLOAT) <= SHMEM_SIZE_MAX 
    srp += cc[((i*4+0)*CC_PAIR_SIZE) + tmp.x];
    srp += cc[((i*4+1)*CC_PAIR_SIZE) + tmp.y];
    srp += cc[((i*4+2)*CC_PAIR_SIZE) + tmp.z];
    srp += cc[((i*4+3)*CC_PAIR_SIZE) + tmp.w];
#else
    srp += d_cc[((i*4+0)*CC_PAIR_SIZE) + tmp.x];
    srp += d_cc[((i*4+1)*CC_PAIR_SIZE) + tmp.y];
    srp += d_cc[((i*4+2)*CC_PAIR_SIZE) + tmp.z];
    srp += d_cc[((i*4+3)*CC_PAIR_SIZE) + tmp.w];
#endif
  }
  d_power[q]= srp;
}




//<<<POWER_SIZE/128,128>>>
__global__ void find_max_by_reduction(const float* d_power, 
                                      int* d_max_indices, 
                                      float* d_max_powers) {
  int bx= blockIdx.x, tx= threadIdx.x; 
#if Q<=97200  
  __shared__ float cache[128];
  if (bx >= (int)(Q/128)) return;
  cache[tx]= d_power[bx*128 + tx];
#elif Q==388800                         //512blk * 1024threads= 524288threads
  __shared__ float cache[1024];         //4B*1024=4096B
  if (bx >= (int)(Q/1024)) return;
  cache[tx]= d_power[bx*1024 + tx];
#endif
  __syncthreads();
  int i=blockDim.x >> 1;
  while (i!=0) {
    if (tx<i) {
      if (cache[tx] < cache[tx+i]) {
        cache[tx]= cache[tx+i];
        cache[tx+i]= tx+i;
      } else {
        cache[tx+i]= tx;
      }
    }
    __syncthreads();
    i= i>>1;
  }
  if (tx==0) { 
    d_max_powers[bx]= cache[0]; 
  }
  __syncthreads();
  if (tx==0) {
    int idx1=0, idx2=0;
#if Q<=97200      
    idx1= cache[1           ];		// 1
    idx2= cache[2      +idx1];		// 2
    idx1= cache[4      +idx2];		// 4
    idx2= cache[8      +idx1];		// 8
    idx1= cache[16     +idx2];		// 16
    idx2= cache[32     +idx1];		// 32
    idx1= cache[64     +idx2];		// 64
    d_max_indices[bx]= idx1;
#elif Q==388800
	  idx1= cache[1           ];		// 1
	  idx2= cache[2      +idx1];		// 2
	  idx1= cache[4      +idx2];		// 4
	  idx2= cache[8      +idx1];		// 8
	  idx1= cache[16     +idx2];		// 16
	  idx2= cache[32     +idx1];		// 32
	  idx1= cache[64     +idx2];		// 64
	  idx2= cache[128    +idx1];		// 128
	  idx1= cache[256    +idx2];		// 256
	  idx2= cache[512    +idx1];		// 512
	  d_max_indices[ bx]= idx2;
#endif
  }	
}

//<<<1,POWER_SIZE/128>>>  //Q==3888,97200
//<<<1,POWER_SIZE/1024>>> //Q==388800
__global__ void find_max_by_reduction32(int* d_max_indices, 
                                        float* d_max_powers, 
                                        int* d_maxIdx) {
  int tx=threadIdx.x, bx=blockIdx.x;
#if Q<=97200  
  __shared__ float cache[POWER_SIZE/128];
#elif Q==388800
  __shared__ float cache[POWER_SIZE/1024];  //512
#endif
  cache[tx]= d_max_powers[tx];
  __syncthreads();
  {
    int i=blockDim.x>>1;
    while (i!=0) {
      if (tx<i) {
        if (cache[tx] < cache[tx+i]) {
          cache[tx]= cache[tx+i];
          cache[tx+i]= tx+i;
        } 
        else {
          cache[tx+i]= tx;
        }
      }
      __syncthreads();
      i = i>>1;
    }
  }
  if (tx==0) { 
    d_max_powers[bx]= cache[0]; 
  }
  __syncthreads();
  if (tx==0) {  
  	int idx1=0, idx2=0;
#if Q==972
    idx1= cache[1           ];						// 1
    idx2= cache[2      +idx1];						// 2
    idx1= cache[4      +idx2];						// 4
    d_maxIdx[ bx]= idx1*(128) + d_max_indices[idx1];
#elif Q==3888
    idx1= cache[1           ];						// 1
    idx2= cache[2      +idx1];						// 2
    idx1= cache[4      +idx2];						// 4
    idx2= cache[8      +idx1];						// 8
    idx1= cache[16     +idx2];						// 16
    d_maxIdx[ bx]= idx1*(128) + d_max_indices[idx1];
#elif Q==32400
    idx1= cache[1           ];						// 1
    idx2= cache[2      +idx1];						// 2
    idx1= cache[4      +idx2];						// 4
    idx2= cache[8      +idx1];						// 8
    idx1= cache[16     +idx2];						// 16
    idx2= cache[32     +idx1];						// 32
    idx1= cache[64     +idx2];						// 64
    idx2= cache[128    +idx1];						// 128
    d_maxIdx[ bx]= idx2*(128) + d_max_indices[idx2];
#elif Q==97200
    idx1= cache[1           ];						// 1
    idx2= cache[2      +idx1];						// 2
    idx1= cache[4      +idx2];						// 4
    idx2= cache[8      +idx1];						// 8
    idx1= cache[16     +idx2];						// 16
    idx2= cache[32     +idx1];						// 32
    idx1= cache[64     +idx2];						// 64
    idx2= cache[128    +idx1];						// 128
    idx1= cache[256    +idx2];						// 256
    idx2= cache[512    +idx1];						// 512
    d_maxIdx[ bx]= idx2*(128) + d_max_indices[idx2];
#elif Q==388800
    idx1= cache[1           ];						// 1
    idx2= cache[2      +idx1];						// 2
    idx1= cache[4      +idx2];						// 4
    idx2= cache[8      +idx1];						// 8
    idx1= cache[16     +idx2];						// 16
    idx2= cache[32     +idx1];						// 32
    idx1= cache[64     +idx2];						// 64
    idx2= cache[128    +idx1];						// 128
    idx1= cache[256    +idx2];						// 256
    d_maxIdx[ bx]= idx1*(1024) + d_max_indices[idx1];
#endif
		// 360000(512), 129600(512), 32400(128), 10800(128)
		// 97200(512), 64800(256), 32400(128), 19440(128), 3888(16), 972(4)
  }	
}


static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

static __device__ __host__ inline Complex ComplexConj(Complex a) {
	Complex c;
	c.x= a.x;
	c.y= -1*a.y;
	return c;
}

static __device__ __host__ inline float ComplexAbs(Complex a) {
	float ret= sqrt((a.x*a.x) + (a.y*a.y));
	return ret;
}


/////////////////////////////////////////////////////////////////////
//////////////////// TABLE LOADER ///////////////////////////////////
int32_t load_TDOA_table(void) {
  int32_t n;
  FILE *fp= fopen("./TDOA_table.bin","rb"); 
  assert(fp!=NULL);
  assert((N%4)==0);
  int32_t tab_idx=0;
  int32_t tab_ch4_idx=0;
  for (int i=0; i<Q; ++i) {
    int16_t tdoa[N];
    int16_t tdoa_char4[4];
    int32_t tdoa_char4_idx=0;
    n= fread(tdoa,sizeof(int16_t),N,fp);
    assert(n==N);
    for (int32_t j=0; j<N; ++j) {
      int16_t biased_tdoa= tdoa[j] + MAX_SD;
      tdoa_char4[tdoa_char4_idx++]= biased_tdoa;
      if (tdoa_char4_idx>=4) {
        tdoa_char4_idx=0;
        h_tdoa_table_char4[tab_ch4_idx].x= tdoa_char4[0];
        h_tdoa_table_char4[tab_ch4_idx].y= tdoa_char4[1];
        h_tdoa_table_char4[tab_ch4_idx].z= tdoa_char4[2];
        h_tdoa_table_char4[tab_ch4_idx].w= tdoa_char4[3];
        ++tab_ch4_idx;
      }
      h_TDOA_table[tab_idx++]= biased_tdoa;
    }
	}
	fclose(fp);
	return 1;
}

int32_t load_TOA_table(void) {
  int32_t n;
  FILE *fp= fopen("./TOA_table.bin","rb");
  assert(fp!=NULL);
  n= fread(h_TOA_table,sizeof(float),Q*M,fp);
  assert(n==Q*M);
  fclose(fp);
  return 1; 
}

int32_t load_sphCoords(void) {
  int32_t n;
  FILE* fp= fopen("./sphCoords.bin","rb");
  assert(fp!=NULL);
  n= fread(h_sphCoords,sizeof(int32_t),Q*3,fp);
  assert(n==Q*3);
  fclose(fp);
  return 1;
}
//////////////////// TABLE LOADER ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

