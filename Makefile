sum:
	#g++ -std=c++11 -pthread cpusum.cpp -o csum -fopenmp
	nvcc --std c++11 sum3.cu -o gsum

clean:
	rm -rf *.o
	rm -rf csum gsum sum
