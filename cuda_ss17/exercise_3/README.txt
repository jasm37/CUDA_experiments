Exercise 3

Check the code in gamma/main.cu. To run it, choose parameters -repeats (integer), -gamma (float) and -gray(to use freyscaled image or not), for example:

	./main -i ../../images/owl.png -repeats 2 -gamma 3.5 -gray


1. 2. Check the code in gamma/main.cu

	OUTPUT:
	In the output of the code it is printed:
	-Avg. CPU time(through all the repeatds)
	-GPU time for (x,y) block configurations (2D block)
	
3. 4.	-Time spent without memory allocation/deletion:
		GPU time for (dim_x=32, dim_y=8) is 0.721 ms
		GPU time for (dim_x=64, dim_y=4) is 0.387 ms
		GPU time for (dim_x=96, dim_y=2) is 0.243 ms
		GPU time for (dim_x=128, dim_y=2) is 0.248 ms
		GPU time for (dim_x=256, dim_y=1) is 0.208 ms
	-Time spent WITH memory allocations/deletions(only kernel):
		GPU time for (dim_x=32, dim_y=8) is 2.061 ms
		GPU time for (dim_x=64, dim_y=4) is 1.629 ms
		GPU time for (dim_x=96, dim_y=2) is 1.475 ms
		GPU time for (dim_x=128, dim_y=2) is 1.59 ms
		GPU time for (dim_x=256, dim_y=1) is 1.491 ms
	3. GPU is considerably faster the CPU and GPU allocations/ deletions are very expensive in comparison with the time the kernel takes to compute the gamma correction.
	4. A linear configuration of threads per block gives better results due to the design of the kernel(it only uses one dimension of the block).
