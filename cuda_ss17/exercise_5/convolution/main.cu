// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#define _USE_MATH_DEFINES
#include "helper.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

void clamp(int &i, int &j, int dimx, int dimy)
{
    if(i<0) i=0;
    else if(i>dimx-1) i= dimx-1;
    if(j<0) j=0;
    else if(j>dimy-1) j= dimy-1;
}

void do_convolution(float *conv, float *ker, float *a, int r, int dimx, int dimy, int nc)
{
    int drad = 2*r+1; // double radius
    int ni, nj;
    double temp = 0;
    for(int c=0; c<nc; c++)
        for(int j = 0; j<dimy; j++)
            for(int i = 0; i<dimx; i++)
            {
                temp = 0;
                for(int ir = 0; ir<drad; ir++)
                    for(int jr = 0; jr<drad; jr++)
                    {
                        ni = i-ir+r;
                        nj = j-jr+r;
                        clamp(ni,nj,dimx,dimy);
                        temp += ker[ir + drad*jr]*a[ni + dimx*nj + dimx*dimy*c];
                    }
                conv[i + dimx*j + dimx*dimy*c] = temp;
            }
}


__device__
void clampGPU(int &i, int &j, int dimx, int dimy)
{
    if(i<0) i=0;
    else if(i>dimx-1) i= dimx-1;
    if(j<0) j=0;
    else if(j>dimy-1) j= dimy-1;
}

__global__
void do_GPUconvolution(float *conv, float *ker, float *a, int r, int dimx, int dimy, int nc)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int n = threadIdx.z + blockDim.z * blockIdx.z;
    int drad = 2*r+1; // double radius
    int ni, nj;
    double temp;
    if( i < dimx && j <dimy && n<nc )
    {
        temp = 0;
        for(int ir = 0; ir<drad; ir++)
            for(int jr = 0; jr<drad; jr++)
            {
                ni = i-ir+r;
                nj = j-jr+r;
                clampGPU(ni, nj, dimx, dimy);
                temp += ker[ir + drad*jr]*a[ni + dimx*nj + dimx*dimy*n];
            }
        conv[i + dimx*j + dimx*dimy*n] = temp;
    }

}


void kernel_comp(float *ker, float std, int radius)
{
    float coeff = 1 / (2 * M_PI * std * std);
    float arg;
    int rad = radius*2+1;
    int in, jn;
    float sum= 0;

    for(int i=0; i<rad; i++)
        for(int j=0; j<rad; j++)
        {   in = i-radius; jn = j-radius;
            arg = -( in*in + jn*jn ) / (2*std*std);
            ker[i + rad*j] = coeff * expf(arg);
            sum += ker[i + rad*j];
        }

    for(int i=0; i<rad; i++)
        for(int j=0; j<rad; j++)
            ker[i + rad*j] /= sum;

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

    float std = 1.f;
    getParam("std", std, argc, argv);
    cout << "std: " << std << endl;

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

    //////////////////////////
    //Question 1&2 Solutions 
    /////////////////////////
    
    Timer timer; timer.start();
    int rad = ceil(3*std);
    int drad = 2*rad+1;
    cv::Mat mKer(drad,drad,CV_32FC1);
    float *ker = new float[drad*drad];
    kernel_comp(ker, std, rad);
    float max = 0.0f;
    for(int i=0; i<drad*drad; i++)
    {
        if(ker[i]>max) max=ker[i];
    }
    for(int i=0; i<drad*drad; i++)
    {
        ker[i] /= max;
    }

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "Kernel construction time: " << t*1000 << " ms" << endl;
    convert_layered_to_mat(mKer, ker);
    showImage("Output Kernel", mKer, 100+w+40, 100);
    
    

    //////////////////////////////////
    //Question 3: Convolution on CPU
    //////////////////////////////////
    
    timer.start();
    //int rad = ceil(3*std);
    drad = 2*rad+1;
    //float *ker = new float[drad*drad];
    kernel_comp(ker, std, rad);
    float *conv = new float[w*h*nc];
    do_convolution(conv, ker, imgIn, rad, w,h,nc);
    timer.end();  t = timer.get();  // elapsed time in seconds
    cout << "CPU convolution time: " << t*1000 << " ms" << endl;
    //showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    cv::Mat mConv(h,w,mIn.type());
    convert_layered_to_mat(mConv, conv);
    showImage("Output CPU", mConv, 100+w+40, 100);
    

    //////////////////////
    //Question 5 Solution
    /////////////////////
    //int rad = ceil(3*std);
    int double_rad = 2*rad+1;
    //float *ker = new float[double_rad*double_rad];
    kernel_comp(ker, std, rad);

    int size_elem = w*h*nc;
    float *d_ker, *d_conv, *d_imgIn, *d_imgOut;
    //float *conv = new float[size_elem];
    size_t nbytes = size_t(size_elem)*sizeof(float);

    int dim_x = 16;
    int dim_y = 4;
    int dim_z = 4;
    // Initialize stuff
    dim3 block = dim3(dim_x,dim_y,dim_z);
    dim3 grid = dim3((w + block.x -1) / block.x, (h + block.y -1) / block.y, (nc + block.z -1) / block.z);

    // Memory allocation
    cudaMalloc(&d_ker, double_rad*double_rad*sizeof(float));CUDA_CHECK;
    cudaMemset(d_ker, 0, double_rad*double_rad*sizeof(float));
    cudaMalloc(&d_imgOut, nbytes);CUDA_CHECK;
    cudaMemset(d_imgOut, 0, nbytes);
    cudaMalloc(&d_conv, nbytes);CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes);CUDA_CHECK;
    cudaMemset(d_imgIn, 0, nbytes);
    cudaMemcpy( d_ker, ker, double_rad*double_rad*sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
    cudaMemcpy( d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice );CUDA_CHECK;

    timer.start();
    do_GPUconvolution<<<grid, block>>>(d_conv, d_ker, d_imgIn, rad, w, h, nc);
    timer.end();  t = timer.get();  // elapsed time in seconds
    cout << "GPU convolution: time: " << t*1000 << " ms" << endl;

    cudaMemcpy( conv, d_conv, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;

    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    //cv::Mat mConv(h,w,mIn.type());
    convert_layered_to_mat(mConv, conv);
    showImage("Output Convolution", mConv, 200, 100);
   
    
    cudaFree(d_imgOut);CUDA_CHECK;
    cudaFree(d_ker);CUDA_CHECK;
    cudaFree(d_conv);CUDA_CHECK;
    cudaFree(d_imgIn);CUDA_CHECK;

    /////////////// END /////////////
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
    //delete[] ker;
    //delete[] conv;
    // close all opencv windows
    cvDestroyAllWindows();

    return 0;
}



