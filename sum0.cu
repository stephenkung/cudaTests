#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <vector>
using namespace std;


#define DSIZE 20000000

bool cuda_init(){
	int count;
	cudaGetDeviceCount(&count);
	cout<<"cuda device count:"<<count<<endl;
	cudaDeviceProp prop;
	if(cudaGetDeviceProperties(&prop,0)==cudaSuccess){
        	printf( "   --- General Information for device %d ---\n", 0 );
        	printf( "Name:  %s\n", prop.name );
        	printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        	printf( "Clock rate:  %d\n", prop.clockRate );
        	printf( "Device copy overlap:  " );
        	if (prop.deviceOverlap)
        	    printf( "Enabled\n" );
        	else
        	    printf( "Disabled\n");
        	printf( "Kernel execution timeout :  " );
        	if (prop.kernelExecTimeoutEnabled)
        	    printf( "Enabled\n" );
        	else
        	    printf( "Disabled\n" );
 
        	printf( "   --- Memory Information for device %d ---\n", 0 );
        	printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        	printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        	printf( "Max mem pitch:  %ld\n", prop.memPitch );
        	printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
 
        	printf( "   --- MP Information for device %d ---\n", 0 );
        	printf( "Multiprocessor count:  %d\n",
        	            prop.multiProcessorCount );
        	printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        	printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        	printf( "Threads in warp:  %d\n", prop.warpSize );
        	printf( "Max threads per block:  %d\n",
        	            prop.maxThreadsPerBlock );
        	printf( "Max thread dimensions:  (%d, %d, %d)\n",
        	            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
        	            prop.maxThreadsDim[2] );
        	printf( "Max grid dimensions:  (%d, %d, %d)\n",
        	            prop.maxGridSize[0], prop.maxGridSize[1],
        	            prop.maxGridSize[2] );
        	printf( "\n" );
	}
	return true;
} 


void initData(int data[]){
	for(int i=0; i<DSIZE; i++){
		data[i] = rand()%100;
	}
}



__global__ void sumGPU(int dataGPU[], int* result /*return val*/){
	//tid: current id of a thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;   
	int sum = 0;	

	for(int i= tid ;  i < DSIZE; i+= gridDim.x*blockDim.x) {  
		sum += dataGPU[i]*dataGPU[i];	
	}
	
	result[tid] = sum;
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
	dim3 blocksize(256);
	dim3 gridsize(16);
	cout<<"block size:"<<blocksize.x<<endl;
	cout<<"grid size:"<<gridsize.x<<endl;
	
	clock_t start, end; start= clock();
	//int dataGPU[DSIZE]; //this is wrong, cpu will allocate memory from stack 
	int* dataGPU;  
	int result_size = blocksize.x*gridsize.x;
	int* result;
	cudaMalloc((void**)&dataGPU,sizeof(int)*DSIZE);
	cudaMalloc((void**)&result, sizeof(int)*result_size); 	

	cudaMemcpy(dataGPU,data,sizeof(int)*DSIZE,cudaMemcpyHostToDevice);
	sumGPU<<<gridsize, blocksize>>>(dataGPU,result);

	int psum [blocksize.x*gridsize.x];
	cudaMemcpy(psum,result,sizeof(int)*blocksize.x*gridsize.x,cudaMemcpyDeviceToHost);
	 
	int sum=0;
	for(int i=0; i<blocksize.x*gridsize.x; i++){
		sum += psum[i];	
	}
	cout<<"GPU result:"<<sum<<endl;
	end=clock();
	cout<<"GPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
}
