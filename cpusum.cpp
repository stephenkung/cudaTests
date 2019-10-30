#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <vector>
#include <math.h>
using namespace std;


#define DSIZE 20000000



void initData(int data[]){
	for(int i=0; i<DSIZE; i++){
		data[i] = rand()%100;
	}
}


//one thread
int sumCPU(int data[]){
	clock_t start, end; start= clock();
	int sum=0;
	for(int i=0; i<DSIZE; i++){
		sum += data[i]^2;
	}
	end=clock();
	cout<<"CPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
	cout<<"CPU one-thread result:"<<sum<<endl;
	return sum;
}


int sumCPUMP(int data[]){
	clock_t start, end; start= clock();
	int sum=0;
	#pragma omp parallel for reduction(+:sum) 
	for(int i=0; i<DSIZE; i++){
		sum += data[i]^2;
	}
	end=clock();
	cout<<"CPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
	cout<<"CPU one-thread result:"<<sum<<endl;
	return sum;
}


void accBlock(int start, int end, int data[], int& res){
	for(int i=start; i<end; i++){
		res += data[i]^2;
	}
}


//multi-thread
int sumCPUMulti(int data[]){
	clock_t start, end; start= clock();
    	unsigned long const hardware_threads = thread::hardware_concurrency();//获取PC的CPU core数目，C++库可能无法访问该信息，所以可能返回0  
	cout<<"num of threads cpu support:"<<hardware_threads<<endl;
    	unsigned long const num_threads =  hardware_threads != 0 ? hardware_threads : 2;  
    	unsigned long const block_size = DSIZE / num_threads;//计算每个线程需要执行的序列大小  
    	vector<thread> threads;
    	int results[num_threads];  for(int i=0; i<num_threads; i++){results[i]=0;} 
    	int block_start = 0;
    	for (int i = 0; i < num_threads; i++){
        	auto block_end = block_start;
        	block_end += block_size;
        	thread thread_temp(accBlock, block_start, block_end, data, ref(results[i]));//result by reference 
		threads.push_back(move(thread_temp)); //must use move
        	block_start = block_end; 
    	}
    	for(auto i=threads.begin(); i<threads.end(); i++){
		i->join();
	}
    	int sum=0;
	accBlock(0, num_threads, results, sum);
	end=clock();
	cout<<"CPU running time(ms):"<<(double)(end-start)*1000.0f/CLOCKS_PER_SEC<<endl;
	cout<<"CPU Multi-thread result:"<<sum<<endl;
	return sum;
	
}




///////////////////////  main program ////////////////
int main(){
	//int data[DSIZE];
	int* data = new int[DSIZE];
	initData(data);
	cout<<"\n*******1 thread******"<<endl;
	sumCPU(data);
	cout<<"\n******multi thread*******"<<endl;
	sumCPUMulti(data);
	cout<<"\n******openMP*******"<<endl;
	sumCPUMP(data);
	return 1;
}
