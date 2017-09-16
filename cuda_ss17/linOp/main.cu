// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA


__global__
void l2norm(float *a, float *b, int dimx, int dimy, int n)
{
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    int norm = 0;
    if(ind < dimx*dimy)
    {
        for(int i = 0; i<n; i++)
        {
            norm += pow(a[ind + i*dimx*dimy], 2);
        }
        b[ind] = sqrtf(norm);
    }

}

__global__
void div(float *div_vec, float *dx_a, float *dy_a, float *dxx_a, float *dyy_a, int dimx, int dimy, int n)
{
    // n is the number of channels
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    // Assumes that dxx_a and dyy_a are initialized as zero arrays
    int sub = ind %(dimx*dimy);
    if(ind < dimx*dimy*n)
    {
        if(sub%dimx == 0) dxx_a[ind]=(dx_a[ind]-dx_a[ind-1]);
        if( sub >= dimx ) dyy_a[ind]=(dy_a[ind]-dy_a[ind-dimx]);
        div_vec[ind] = dxx_a[ind] + dyy_a[ind];
    }
}

__global__
void d_plus(float *a, float *dx_a, float *dy_a, int dimx, int dimy, int n)
{
    // n is the number of channels
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    // Assumes that dx_a and dy_a are initialized as zero arrays
    int sub = ind %(dimx*dimy);
    if(ind < dimx*dimy*n)
    {
        if( (sub+1)%dimx != 0 ) dx_a[ind]=(a[ind+1]-a[ind]);
        if( sub <= dimy*(dimx-1) ) dy_a[ind]=(a[ind+dimx]-a[ind]);
    }
}



int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];
    //float *imgOut = new float[(size_t)w*h];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    //Timer timer; timer.start();
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###
    int size_elem = w*h*nc;
    float *d_imgOut, *dx_a, *dy_a, *dxx_a, *dyy_a, *div_vec;
    float *result = new float[(size_t)w*h];
    size_t nbytes = size_t(size_elem)*sizeof(float);

    int dim_x = 32;
    int dim_y = 8;
    // Initialize stuff
    dim3 block = dim3(dim_x,dim_y,1);
    dim3 grid = dim3((size_elem + block.x -1) / block.x, 1, 1);

    cudaMalloc(&d_imgOut, nbytes);CUDA_CHECK;
    cudaMemset(d_imgOut, 0, nbytes);
    cudaMalloc(&dx_a, nbytes);CUDA_CHECK;
    cudaMemset(dx_a, 0, nbytes);
    cudaMalloc(&dy_a, nbytes);CUDA_CHECK;
    cudaMemset(dy_a, 0, nbytes);
    cudaMalloc(&dxx_a, nbytes);CUDA_CHECK;
    cudaMemset(dxx_a, 0, nbytes);
    cudaMalloc(&dyy_a, nbytes);CUDA_CHECK;
    cudaMemset(dyy_a, 0, nbytes);
    cudaMalloc(&div_vec, w*h*sizeof(float));CUDA_CHECK;
    cudaMemset(div_vec, 0, w*h*sizeof(float));
    //cudaMalloc(&result, w*h*sizeof(float));CUDA_CHECK;
    //cudaMemset(result, 0, w*h*sizeof(float));CUDA_CHECK;

    cudaMemcpy( d_imgOut, imgIn, nbytes, cudaMemcpyHostToDevice );CUDA_CHECK;

    //Call function
    Timer mTimer;mTimer.start();
    //float *a, float *dx_a, float *dy_a, int dimx, int dimy, int n)
    // n is the number of channels

    d_plus<<<grid, block>>>(d_imgOut, dx_a, dy_a, w, h, nc);
    div<<<grid, block>>>(div_vec, dx_a, dy_a, dxx_a, dyy_a, w, h, nc);
    CUDA_CHECK;

    //d_plus<<<grid, block>>>(d_imgOut, dx_a, dy_a, w, h, nc);
    //mTimer.end();  t = mTimer.get();  // elapsed time in seconds
    //l2norm<<<grid, block>>>(div_vec, result, w, h, nc);
    cudaMemcpy( result, div_vec, w*h*sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;
    cv::Mat mRes(h,w,CV_32FC3);
    convert_layered_to_mat(mRes, result);
    //cudaMemcpy( imgOut, dx_a, w*h*nc*sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;

    cudaFree(d_imgOut);CUDA_CHECK;
    cudaFree(dx_a);CUDA_CHECK;
    cudaFree(dy_a);CUDA_CHECK;
    cudaFree(dxx_a);CUDA_CHECK;
    cudaFree(dyy_a);CUDA_CHECK;
    cudaFree(div_vec);CUDA_CHECK;
    cudaFree(result);CUDA_CHECK;
    //timer.end();  float t = timer.get();  // elapsed time in seconds
    //cout << "time: " << t*1000 << " ms" << endl;






    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mRes, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



