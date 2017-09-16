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


__device__
void clampGPU(int &i, int &j, int dimx, int dimy)
{
    if(i<0) i=0;
    else if(i>dimx-1) i= dimx-1;
    if(j<0) j=0;
    else if(j>dimy-1) j= dimy-1;
}

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
    if(ind < dimx*dimy)
    {
        if(sub%dimx == 0) dxx_a[ind]=(dx_a[ind]-dx_a[ind-1]);
        if( sub >= dimx ) dyy_a[ind]=(dy_a[ind]-dy_a[ind-dimx]);
        div_vec[ind] = dxx_a[ind] + dyy_a[ind];
    }
}

__global__
void d_plus_rot(float *a, float *d_grad, float *dx_a, float *dy_a, int dimx, int dimy, int nc)
{
    // n is the number of channels
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    // Assumes that dx_a and dy_a are initialized as zero arrays
    //int sub = ind %(dimx*dimy);
    float cte = 1.0f/32.0f;
    float temp_dx = 0, temp_dy = 0;
    if(i < dimx && j < dimy && k < nc)
    {
        int temp_pi = i+1, temp_pj=j+1, temp_ni=i-1, temp_nj=j-1;
        if(i==0) temp_ni = 0;
        else if(i==dimx-1) temp_pi = dimx-1;
        if(j==0) temp_nj=0;
        else if(j==dimy-1) temp_pj = dimy-1;

        temp_dx=cte*(
                    3*a[temp_pi + dimx*(temp_pj) + dimx*dimy*k] +
                    10*a[temp_pi + dimx*j + dimx*dimy*k] +
                    3*a[temp_pi + dimx*temp_nj + dimx*dimy*k]
                    -(3*a[temp_ni + dimx*temp_pj + dimx*dimy*k] +
                    10*a[temp_ni + dimx*j + dimx*dimy*k] +
                    3*a[temp_ni + dimx*temp_nj + dimx*dimy*k]));
        temp_dy=cte*(
                    3*a[temp_pi + dimx*(temp_pj) + dimx*dimy*k] +
                    10*a[i + dimx*temp_pj + dimx*dimy*k] +
                    3*a[temp_ni + dimx*temp_pj + dimx*dimy*k]
                    -(3*a[temp_pi + dimx*temp_nj + dimx*dimy*k] +
                    10*a[i + dimx*temp_nj + dimx*dimy*k] +
                    3*a[temp_ni + dimx*temp_nj + dimx*dimy*k]));
        dx_a[i+dimx*j+dimx*dimy*k] = temp_dx;
        dy_a[i+dimx*j+dimx*dimy*k] = temp_dy;
        d_grad[i+dimx*j+dimx*dimy*k] = temp_dx + temp_dy;
    }
}

__global__
void d_plus(float *a, float *d_grad, int dimx, int dimy, int nc)
{
    // n is the number of channels
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    // Assumes that dx_a and dy_a are initialized as zero arrays
    //int sub = ind %(dimx*dimy);
    float tempx=0.0f, tempy=0.0f;
    if(i<dimx && j<dimy && k<nc)
    {
        int index = i+dimx*j+dimx*dimy*k;
        if( i<dimx-1 ) tempx= (a[index+1]-a[index]);
        if( j<dimy-1 ) tempy=(a[index+dimx]-a[index]);
        //d_grad[index] = tempx + tempy;
        d_grad[index] = tempx+tempy;
    }
}

__global__
void compute_M(float *joint, float *coord_1, float *coord_2, float *coord_3, float *dx_a, float *dy_a, int dimx, int dimy, int nc)
{
    // n is the number of channels
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    int ind, ind2, ind3;
    float temp1 = 0, temp2 = 0, temp3 = 0;
    if (i < dimx && j < dimy && k == 0 )
    {
        for (int n = 0; n < nc; n++)
        {
            ind = i+dimx*j+dimx*dimy*n;
            temp1 += dx_a[ind]*dx_a[ind];
            temp2 += dx_a[ind]*dy_a[ind];
            temp3 += dy_a[ind]*dy_a[ind];
        }

        ind2 = i+dimx*j;
        coord_1[ind2] = temp1;
        coord_2[ind2] = temp2;
        coord_3[ind2] = temp3;


        // Store joint matrix M for convolution
        ind3 = 4*ind2;
        joint[ind3] = temp1;
        joint[1+ind3] = temp2;
        joint[2+ind3] = temp2;
        joint[3+ind3] = temp3;
    }
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


    float std = 1;
    int rad = ceil(3*std);
    int double_rad = 2*rad+1;
    float *ker = new float[double_rad*double_rad];
    kernel_comp(ker, std, rad);

    int size_elem = w*h*nc;
    //float *d_imgOut, *dx_a, *dy_a, *dxx_a, *dyy_a, *div_vec, *result = NULL;

    float *d_ker, *d_conv, *d_imgIn, *d_imgOut, *d_grad, *d_dx_conv, *d_dy_conv, *d_coeff1, *d_coeff2, *d_coeff3, *d_coeff_joint, *d_coeff_temp;
    float *dd_coeff1, *dd_coeff2, *dd_coeff3;
    float *m_coord1 = new float[w*h], *m_coord2 = new float[w*h], *m_coord3 = new float[w*h];
    float *dx_a = new float[size_elem];
    float *coeff_joint = new float[4*w*h];
    float *conv = new float[size_elem];

    size_t nbytes = size_t(size_elem)*sizeof(float);

    int dim_x = 16;
    int dim_y = 4;
    int dim_z = 4;
    // Initialize stuff
    dim3 block = dim3(dim_x,dim_y,dim_z);
    dim3 grid = dim3((w + block.x -1) / block.x, (h + block.y -1) / block.y, (nc + block.z -1) / block.z);

    cudaMalloc(&d_ker, double_rad*double_rad*sizeof(float));CUDA_CHECK;
    cudaMemset(d_ker, 0, double_rad*double_rad*sizeof(float));
    cudaMalloc(&d_imgOut, nbytes);CUDA_CHECK;
    cudaMemset(d_imgOut, 0, nbytes);
    cudaMalloc(&d_grad, nbytes);CUDA_CHECK;
    cudaMemset(d_grad, 0, nbytes);
    cudaMalloc(&d_dx_conv, nbytes);CUDA_CHECK;
    cudaMemset(d_dx_conv, 0, nbytes);
    cudaMalloc(&d_dy_conv, nbytes);CUDA_CHECK;
    cudaMemset(d_dy_conv, 0, nbytes);
    cudaMalloc(&d_conv, nbytes);CUDA_CHECK;
    //cudaMemset(d_conv, 0, nbytes);
    cudaMalloc(&d_imgIn, nbytes);CUDA_CHECK;
    cudaMemset(d_imgIn, 0, nbytes);
    cudaMemcpy( d_ker, ker, double_rad*double_rad*sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
    cudaMemcpy( d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice );CUDA_CHECK;
    size_t coeff_size = 4*w*h*sizeof(float);
    cudaMalloc(&d_coeff_joint, coeff_size);CUDA_CHECK;
    cudaMemset(d_coeff_joint, 0, nbytes);

    cudaMalloc(&d_coeff_temp, coeff_size);CUDA_CHECK;
    cudaMemset(d_coeff_temp, 0, nbytes);



    size_t grid_bytes = size_t(w*h)*sizeof(float);
    cudaMalloc(&d_coeff1, grid_bytes);CUDA_CHECK;
    cudaMemset(d_coeff1, 0, grid_bytes);
    cudaMalloc(&d_coeff2, grid_bytes);CUDA_CHECK;
    cudaMemset(d_coeff2, 0, grid_bytes);
    cudaMalloc(&d_coeff3, grid_bytes);CUDA_CHECK;
    cudaMemset(d_coeff3, 0, grid_bytes);

    cudaMalloc(&dd_coeff1, grid_bytes);CUDA_CHECK;
    cudaMalloc(&dd_coeff2, grid_bytes);CUDA_CHECK;
    cudaMalloc(&dd_coeff3, grid_bytes);CUDA_CHECK;

    Timer timer; timer.start();


    do_GPUconvolution<<<grid, block>>>(d_conv, d_ker, d_imgIn, rad, w, h, nc);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;


    //cudaMemcpy( conv, d_conv, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    //void d_plus_rot(float *a, float *dx_a, float *dy_a, int dimx, int dimy, int nc)
    d_plus_rot<<<grid, block>>>(d_conv, d_grad, d_dx_conv, d_dy_conv, w, h,nc);
    //d_plus(float *d_conv, float *d_grad, int dimx, int dimy, int nc
    //d_plus<<<grid, block>>>(d_conv, d_grad, w, h, nc );
    //cudaMemcpy( dx_a, d_dx_conv, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    //cudaMemcpy( dx_a, d_grad, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    //void compute_M(float *coord_1, float *coord_2, float *coord_3, float *dx_a, float *dy_a, int dimx, int dimy, int nc)


    compute_M<<<grid, block>>>(d_coeff_joint, d_coeff1, d_coeff2, d_coeff3, d_dx_conv, d_dy_conv, w, h, nc);
    cudaMemcpy( m_coord1, d_coeff1, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    cudaMemcpy( m_coord2, d_coeff2, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    cudaMemcpy( m_coord3, d_coeff3, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;




    //do_GPUconvolution<<<grid, block>>>(d_coeff_temp, d_ker, d_coeff_joint, rad, 2, 2, w*h);
    //cudaMemcpy( coeff_joint, d_coeff_temp, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    do_GPUconvolution<<<grid, block>>>(dd_coeff1, d_ker, d_coeff1, rad, w, h, 1);
    cudaMemcpy( m_coord1, dd_coeff1, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;

    do_GPUconvolution<<<grid, block>>>(dd_coeff2, d_ker, d_coeff2, rad, w, h, 1);
    cudaMemcpy( m_coord2, dd_coeff2, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;

    do_GPUconvolution<<<grid, block>>>(dd_coeff3, d_ker, d_coeff3, rad, w, h, 1);
    cudaMemcpy( m_coord3, dd_coeff3, grid_bytes, cudaMemcpyDeviceToHost );CUDA_CHECK;

    float scale = 10.f;
    cv::Mat mCoord1(h,w,CV_32FC1);
    convert_layered_to_mat(mCoord1, m_coord1);
    showImage("C1_out", scale*mCoord1, 100+w+10, 100);
    cv::Mat mCoord2(h,w,CV_32FC1);
    convert_layered_to_mat(mCoord2, m_coord2);
    showImage("C2_out", scale*mCoord2, 100+w+20, 100);
    cv::Mat mCoord3(h,w,CV_32FC1);
    convert_layered_to_mat(mCoord3, m_coord3);
    showImage("C3_out", scale*mCoord3, 100+w+30, 100);



    //cudaMemcpy( imgOut, d_dy_conv, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    cudaMemcpy( imgOut, d_grad, nbytes, cudaMemcpyDeviceToHost );CUDA_CHECK;
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    /*
    cv::Mat mConv(h,w,mIn.type());
    convert_layered_to_mat(mConv, imgOut);
    showImage("Output", mConv, 100+w+40, 100);
    */


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
    //delete[] ker;
    //delete[] conv;
    // close all opencv windows
    cvDestroyAllWindows();

    return 0;
}


