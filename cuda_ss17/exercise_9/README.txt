Exercise 9
The code is in iso_diff/main.cu
TO run the code set -nit, -eps, -tau and -type. -type refers to the type of diffusivity scalar to use, as in question 8: 0 means constant g = 1, 1 means g=inv(max(eps,s)) and 2 means g= exp(-s...). For example:
	./main -i ../../images/owl.png -nit 2 -eps 0.01 -tau 0.01 -type 0

6. If tau is chosen too big the procedure diverges and the image is not and adequate result. If Tau is small and N is big, we will notice a difference image depending on the type of diffusitivity function we picked.

7. If g=1 then the results are similar since they both are gaussian diffusions. To compare, run the convolution code and compare.

8. The results resemble the outputs of the Huber diffusitivity and a similar one where the edge are distorted.

