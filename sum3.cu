#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <vector>
using namespace std;


#define DSIZE 20000000
#define BSIZE 512


void initData(int data[]){
	for(int i=0; i<DSIZE; i++){
		data[i] = rand()%10;
	}
}



/*
use shared memory to do reduciton
multiple threads access shared mem
no dymamic thread number
*/
/*
__global__ void sumGPU(int dataGPU[], int* result){
	//tid: current id of a thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;   
	
	__shared__ int res[BSIZE];  //shared memory inside a block

	res[threadIdx.x] = 0;

	for(int i= tid ;  i < DSIZE; i+= gridDim.x*blockDim.x) {  
		res[threadIdx.x] += dataGPU[i]*dataGPU[i];	
	}
	__syncthreads();

	//for loop unroll
	__shared__ int res1[32];  //shared memory inside a block
	if(threadIdx.x < 32){ 
		res1[threadIdx.x]=0;  //shared memory inside a block
		for(int i=threadIdx.x; i<BSIZE; i += 32){
			res1[threadIdx.x] += res[i];	
			
		}
		//__syncthreads();
	}
	
	if(threadIdx.x ==0){  //one thread only
		for(int i=0; i<32; i++){
			result[blockIdx.x] += res1[i]; 	
		}
	}
}
*/


/*
use dynamic num of threads to do reduction
*/

__global__ void sumGPU(int dataGPU[], int* result){
	//tid: current id of a thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;   
	
	__shared__ int res[BSIZE];  //shared memory inside a block

	res[threadIdx.x] = 0;

	for(int i= tid ;  i < DSIZE; i+= gridDim.x*blockDim.x) {  
		res[threadIdx.x] += dataGPU[i]*dataGPU[i];	
	}
	__syncthreads();

	
	int thread_num = BSIZE/2;
	while(threadIdx.x<thread_num){
		res[threadIdx.x] += res[threadIdx.x+thread_num];		
		__syncthreads();
		thread_num = thread_num/2;
	}
	result[blockIdx.x] = res[0];
}

//one thread
int sumCPU(int data[]){
	clock_t start, end; start= clock();
	int sum=0;
	for(int i=0; i<DSIZE; i++){
		sum += data[i]*data[i];
	}
	end=clock();
	cout<<"CPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
	cout<<"CPU one-thread result:"<<sum<<endl;
	return sum;
}

///////////////////////  main program ////////////////
int main(){
	//cuda_init();
	int* data = new int[DSIZE];
	initData(data);
	cout<<"\n******** CPU ***********"<<endl;
	sumCPU(data);

	/////// GPU job //////////
	cout<<"\n******** GPU ***********"<<endl;
	dim3 blocksize(512);
	dim3 gridsize(16);
	cout<<"block size:"<<blocksize.x<<endl;
	cout<<"grid size:"<<gridsize.x<<endl;
	
	clock_t start, end; start= clock();
	//int dataGPU[DSIZE]; //this is wrong, cpu will allocate memory from stack 
	int* dataGPU;  
	int result_size = gridsize.x;
	int* result;
	cudaMalloc((void**)&dataGPU,sizeof(int)*DSIZE);
	cudaMalloc((void**)&result, sizeof(int)*result_size); 	
	

	cudaMemcpy(dataGPU,data,sizeof(int)*DSIZE,cudaMemcpyHostToDevice);
	sumGPU<<<gridsize, blocksize>>>(dataGPU,result);

	int psum [result_size];
	for(int i=0; i<result_size; i++){psum[i]=0;}
	cudaMemcpy(psum,result,sizeof(int)*result_size,cudaMemcpyDeviceToHost);
	 
	int sum=0;
	for(int i=0; i<result_size; i++){
		sum += psum[i];	
	}
	cout<<"GPU result:"<<sum<<endl;
	end=clock();
	cout<<"GPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
}
