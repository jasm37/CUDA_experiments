Exercise 5
The code is in convolution/main.cu
To run it use -gray and -std to set the std deviation, for example:
	./main -i ../../images/owl.png -std 2.4 -gray

Remarks:	-In main.cu, starting from line 227, there are sections for each subquestion
		-The code will always show the convolution produced the the CPU and GPU 
		-The kernel is alwas shown too
1. 2. 3. 4. Run the code as indicated above
5. Images get blurrier when std is incremented. Also it takes more time as std gets larger because of the number of computation required to compute the convolution(sums) 
