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
using namespace std;

__global__
void convolutionkernel(float *conv, float *ker,float *a, int rad, int w, int h, int nc) 
{
    extern __shared__ float sh_data[];
    //Block indices
    int block_x = blockDim.x * blockIdx.x;
    int block_y = blockDim.y * blockIdx.y;

    // Global indices
    int x = threadIdx.x + block_x;
    int y = threadIdx.y + block_y;

    // Shared memory dimensions
    int sh_w = blockDim.x + 2 * rad;
    int sh_h = blockDim.y + 2 * rad;
    int sh_size = sh_w * sh_h;

    // Division coefficients
    int block_size = blockDim.x * blockDim.y;
    int quot = sh_size / block_size;
    int rest = (sh_size % block_size);
    int local_ind = threadIdx.x + blockDim.x * threadIdx.y;

    // double radius +1
    int drad = 2 * rad + 1; 
    if (local_ind == 0){
        quot = quot + rest;
        rest = 0;
    }

    int pos, xi, yi;
    float temp = 0.f;
    // Repeat per channel
    for (int z = 0; z < nc; z++) 
    { 
        for (int i=0; i<quot; i++)
        {
            pos = local_ind*quot + rest + i;
            // Local coordinates
            xi = (pos%sh_w) + (block_x-rad); 
            yi = (pos/sh_w) + (block_y-rad);
            // Local clamping
            xi = max(0, min(w-1,xi)); 
            yi = max(0, min(h-1,yi));
            // Assign shared mem
            sh_data[pos] = a[xi + (size_t)w*yi + w*h*z];
        }
        
        __syncthreads();
        
        // Do convolution
        if (x < w && y < h) 
        {
            int sh_index = 0;
            temp = 0.f;
            for (int ii = -rad; ii < rad + 1; ii++) 
            {
                for (int jj = -rad; jj < rad + 1; jj++) 
                {
                    sh_index = (threadIdx.x + ii + rad) + (threadIdx.y + jj + rad) *( blockDim.x + 2*rad);
                    temp += sh_data[sh_index] *  ker[(ii + rad + (drad) * (jj + rad))];
                }
            }

            __syncthreads();
            
            conv[x + y * w + z * w * h] = temp;

        }

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


// uncomment to use the camera
//#define CAMERA

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


    // ###
    // ### TODO: Main computation

    // Compute kernel
    float std = 10;
    int rad = ceil(3*std);
    int double_rad = 2*rad+1;
    float *ker = new float[double_rad*double_rad];
    kernel_comp(ker, std, rad);

    // Initialize size variables
    int size_elem = w*h*nc;
    float *d_ker, *d_conv, *d_imgIn, *d_imgOut;
    float *conv = new float[size_elem];
    size_t nbytes = size_t(size_elem)*sizeof(float);

    int dim_x = 16;
    int dim_y = 4;
    int dim_z = 4;



    // Initialize stuff
    dim3 block = dim3(dim_x,dim_y,dim_z);
    dim3 grid = dim3((w + block.x -1) / block.x, (h + block.y -1) / block.y, (nc + block.z -1) / block.z);

    // memory allocation
    cudaMalloc(&d_ker, double_rad*double_rad*sizeof(float));CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes);CUDA_CHECK;
    cudaMalloc(&d_conv, nbytes);CUDA_CHECK;
    cudaMemset(d_conv, 0, nbytes);CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes);CUDA_CHECK;

    cudaMemcpy( d_ker, ker, double_rad*double_rad*sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
    cudaMemcpy( d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice );CUDA_CHECK;
    
    Timer timer; timer.start();
    size_t shmBytes = (dim_x+2*rad)*(dim_y+2*rad)*sizeof(float);
    convolutionkernel<<<grid, block, shmBytes>>>(d_conv, d_ker, d_imgIn, rad, w, h, nc);CUDA_CHECK;
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;
    
    cudaMemcpy( conv, d_conv, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    cv::Mat mConv(h,w,mIn.type());
    convert_layered_to_mat(mConv, conv);
    showImage("Output", mConv, 100+w+40, 100);

    cudaFree(d_imgOut);CUDA_CHECK;
    cudaFree(d_ker);CUDA_CHECK;
    cudaFree(d_conv);CUDA_CHECK;
    cudaFree(d_imgIn);CUDA_CHECK;

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



