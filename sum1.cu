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
__global__ void sumGPU(int dataGPU[], int* result){
	//tid: current id of a thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;   
	int sum = 0;	

	for(int i= tid ;  i < DSIZE; i+= gridDim.x*blockDim.x) {  
		sum += dataGPU[i]*dataGPU[i];	
	}
	
	result[tid] = sum;
}
*/


/*
use shared memory, only one thread access shared mem
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

	unsigned int sum=0;
	if(threadIdx.x==0){  //use one thread to accumulate share memory result
		for(int i=0; i<BSIZE; i++){
			sum += res[i];	
		}
		//printf("sum inside a block:%d \n",sum);
	result[blockIdx.x] = sum;
	}
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
	cudaMemcpy(psum,result,sizeof(int)*result_size,cudaMemcpyDeviceToHost);
	 
	int sum=0;
	for(int i=0; i<result_size; i++){
		sum += psum[i];	
	}
	cout<<"GPU result:"<<sum<<endl;
	end=clock();
	cout<<"GPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
}
