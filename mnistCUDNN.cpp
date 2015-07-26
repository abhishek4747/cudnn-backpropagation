/**
* Copyright 2014 Abhishek Kumar.  All rights reserved.
*
* Author: Abhishek Kumar <abhishek.iitd16@gmail.com>
* 
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasDgemv is used to implement fully connected layers.
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iomanip>

#include <cublas_v2.h>
#include <cudnn.h>

#include "ImageIO.h"

#define value_type double

#define IMAGE_H 28
#define IMAGE_W 28
#define N (IMAGE_H*IMAGE_W)  // dimension of training data

#define DEBUG 0


const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define mymin(a,b) (a>b?b:a)

#define EXIT_WAIVED 0

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status <<"\nError String: "<< cudaGetErrorString(cudaGetLastError());                            \
      FatalError(_error.str());                                        \
    }                                                                  \
}

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

void printHostVector(std::string str, int size, value_type* vec){
	std::cout << str ;
    for (int i = 0; i < mymin(size,400); i++)
    {
        std::cout << std::setprecision(2) << vec[i] << " ";
    }
    std::cout << std::endl; 
}


void printDeviceVector(std::string str, int size, value_type* vec_d)
{
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
	printHostVector(str, size, vec);
    delete [] vec;
}


struct Layer_t
{
    int inputs;
    int outputs;
    int kernel_dim; 	// linear dimension (i.e. size is kernel_dim * kernel_dim)
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
	value_type *output_h, *output_d;
	value_type *del_h, *del_d;

    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0){};

	// Initialize Layers reading weights from the file
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readBinaryFile(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, &data_h, &data_d);
        readBinaryFile(bias_path.c_str(), outputs, &bias_h, &bias_d);
	}
	
	// Initialize Empty Layers
	Layer_t(int _inputs, int _outputs):inputs(_inputs), outputs(_outputs), kernel_dim(1){
		int size = inputs*outputs*kernel_dim*kernel_dim;
		int size_b = outputs;

		int size_ac = size*sizeof(value_type);
		int size_b_ac = size_b*sizeof(value_type);
		int size_o_ac = outputs*sizeof(value_type);
		data_h = new value_type[size];
		bias_h = new value_type[size_b];
		output_h = new value_type[outputs];
		del_h = new value_type[outputs];

		// Random Initialization
		for (int i=0; i<size; i++)
			data_h[i] = (((value_type)rand())/(rand()+1))/100000;
		for (int i=0; i<size_b; i++)
			bias_h[i] = (((value_type)rand())/(rand()+1))/100000;			
		for (int i=0; i<outputs; i++){
			output_h[i]=0;
			del_h[i]=0;
		}
        
		checkCudaErrors( cudaMalloc(&data_d, size_ac) );
        checkCudaErrors( cudaMalloc(&bias_d, size_b_ac) );
        checkCudaErrors( cudaMalloc(&output_d, size_o_ac) );
        checkCudaErrors( cudaMalloc(&del_d, size_o_ac) );

        copyDataToDevice();

		if (DEBUG){
			printHostVector("Weights:\n",size, data_h);
			printHostVector("Bias:\n",size_b, bias_h);
		}
	};
    
    void copyDataToDevice(){
        int size = inputs*outputs*kernel_dim*kernel_dim;
		int size_b = outputs;

		int size_ac = size*sizeof(value_type);
		int size_b_ac = size_b*sizeof(value_type);
		int size_o_ac = outputs*sizeof(value_type);
		
        checkCudaErrors( cudaMemcpy(data_d, data_h, size_ac, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(bias_d, bias_h, size_b_ac, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(output_d, output_h, size_o_ac, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(del_d, del_h, size_o_ac, cudaMemcpyHostToDevice) );
    }
    
    void copyDataToHost(){
        int size = inputs*outputs*kernel_dim*kernel_dim;
		int size_b = outputs;

		int size_ac = size*sizeof(value_type);
		int size_b_ac = size_b*sizeof(value_type);
		int size_o_ac = outputs*sizeof(value_type);
		
        checkCudaErrors( cudaMemcpy(data_h, data_d, size_ac, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(bias_h, bias_d, size_b_ac, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(output_h, output_d, size_o_ac, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(del_h, del_d, size_o_ac, cudaMemcpyDeviceToHost) );
    }



    ~Layer_t()
    {
        delete [] data_h;
		delete [] bias_h;
		delete [] output_h;
		delete [] del_h;
        checkCudaErrors( cudaFree(data_d) );
        checkCudaErrors( cudaFree(bias_d) );
        checkCudaErrors( cudaFree(output_d) );
        checkCudaErrors( cudaFree(del_d) );
    }
private:
    void readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
        std::stringstream error_s;
        if (!dataFile)
        {
            error_s << "Error opening file " << fname; 
            FatalError(error_s.str());
        }
        int size_b = size*sizeof(value_type);
        *data_h = new value_type[size];
        if (!dataFile.read ((char*) *data_h, size_b)) 
        {
            error_s << "Error reading file " << fname; 
            FatalError(error_s.str());
        }
        checkCudaErrors( cudaMalloc(data_d, size_b) );
        checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                    size_b,
                                    cudaMemcpyHostToDevice) );
    }
	void readBinFile(const char* fname, char** data_h, char** data_d, int &m, int n){
		std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
		std::stringstream error_s;
		
		std::cout<<"size of void* "<<sizeof(void*)<<"\t size of int "<<sizeof(int)<<"\t size of float "<<sizeof(float)<<"\t size of short"<<sizeof(short)<<"\t size of double "<<sizeof(double)<<std::endl;
		if (!dataFile){
			error_s << "Error opening file " << fname;
			FatalError(error_s.str());
		}
		
		dataFile.seekg(0, std::ios::end);
		size_t size_b = static_cast<std::string::size_type>(dataFile.tellg());
		dataFile.seekg(0, std::ios::beg);		

        int size = size_b/sizeof(char);
        *data_h = new char[size_b];
        if (!dataFile.read ((char*) *data_h, size_b)) 
        {
            error_s << "Error reading file " << fname; 
            FatalError(error_s.str());
        }
		std::cout << "Read file " << fname << " " << size << " bytes and " << size_b << " value_type " << std::endl; 
		m = size_b/n;

	}
};

class network_t
{
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc, srcDiffTensorDesc, dstDiffTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcDiffTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstDiffTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );

        checkCudaErrors( cublasCreate(&cublasHandle) );
    }
    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcDiffTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstDiffTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCudaErrors( cublasDestroy(cublasHandle) );
    }
  public:
    network_t()
    {
        dataType = CUDNN_DATA_DOUBLE;
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t& layer, int c, value_type *data)
    {
        checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                1, c,
                                                1,
                                                1) );
        value_type alpha = value_type(1);
        value_type beta  = value_type(1);
        checkCUDNN( cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
                                      &alpha, biasTensorDesc,
                                      layer.bias_d,
                                      &beta,
                                      dstTensorDesc,
                                      data) );
    }
    void fullyConnectedForward(const Layer_t& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        resize(dim_y, dstData);

        value_type alpha = value_type(1), beta = value_type(1);
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        checkCudaErrors( cublasDgemv(cublasHandle, CUBLAS_OP_T,
                                      dim_x, dim_y,
                                      &alpha,
                                      ip.data_d, dim_x,
                                      srcData, 1,
                                      &beta,
                                      *dstData, 1) );

        h = 1; w = 1; c = dim_y;
    }
	
	void fullyConnectedBackward(const Layer_t& current_layer, const value_type* last_input){
        int dim_x = current_layer.inputs;
        int dim_y = current_layer.outputs;
		int dim_z = 1;
		value_type* dstData = NULL;
        resize(dim_x*dim_y, &dstData);

        value_type alpha = value_type(1), beta = value_type(0);
        // checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, ip.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		//if (DEBUG) printDeviceVector("last_input: \n", current_layer.inputs, last_input);
		//if (DEBUG) printDeviceVector("del_W: \n", current_layer.outputs, current_layer.del_d);
		
        checkCudaErrors( cublasDgemm(cublasHandle, 
									  CUBLAS_OP_N, CUBLAS_OP_N,
                                      dim_x, dim_y, dim_z,
                                      &alpha,
                                      last_input, dim_x,
                                      current_layer.del_d, dim_z,
                                      &beta,
                                      dstData, dim_x) );
		
		if (DEBUG) printDeviceVector("\tdelta_W (del_W*hidden_input): \n", current_layer.inputs*current_layer.outputs, dstData);

		alpha = value_type(0.1); // learning rate
		beta = value_type(1); 
		//checkCudaErrors( cublasDscal(cublasHandle, ip.inputs*ip.outputs, &alpha, ip.data_d, 1); 
		const value_type* B = current_layer.data_d;
		// C = α op ( A ) + β * C
		// C = 0.1 * delta_W2 + C
		if (DEBUG) printDeviceVector("\tW = W + 0.1*delta_W: old\n", dim_x*dim_y, current_layer.data_d);
		
		checkCudaErrors( cublasDgeam(cublasHandle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										dim_x, dim_y,
										&alpha,
										dstData, dim_x,
										&beta,
										B, dim_x,
										current_layer.data_d, dim_x) );
		if (DEBUG) printDeviceVector("\tW: \n", dim_x*dim_y, current_layer.data_d);

        // place bias into dstData
		dim_x = 1;
		const value_type* B2 = current_layer.bias_d;
		if (DEBUG) printDeviceVector("\tdel_W:\n", current_layer.outputs, current_layer.del_d);
		if (DEBUG) printDeviceVector("\tB = B + 0.1*del_W: old\n", current_layer.outputs, current_layer.bias_d);
		checkCudaErrors( cublasDgeam(cublasHandle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										dim_x, dim_y,
										&alpha,
										current_layer.del_d, dim_x,
										&beta,
										B2, dim_x,
										current_layer.bias_d, dim_x) );
		if (DEBUG) printDeviceVector("\tB:\n", current_layer.outputs, current_layer.bias_d);

		checkCudaErrors( cudaFree(dstData));
    }

    void convoluteForward(const Layer_t& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;

        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h, w) );

        checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
                                              dataType,
                                              conv.outputs,
                                              conv.inputs, 
                                              conv.kernel_dim,
                                              conv.kernel_dim) );
 
        checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                                                   // srcTensorDesc,
                                                    //filterDesc,
                                                    0,0, // padding
                                                    1,1, // stride
                                                    1,1, // upscale
                                                    CUDNN_CROSS_CORRELATION) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                &n, &c, &h, &w) );

        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                0,
                                                &algo
                                                ) );
        resize(n*c*h*w, dstData);
        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                2, 2, // window
                                                0, 0, // padding
                                                2, 2  // stride
                                                ) );
        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w ) );
        h = h / 2; w = w / 2;
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        resize(n*c*h*w, dstData);
        value_type alpha = value_type(1);
        value_type beta = value_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);
        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            CUDNN_ACTIVATION_SIGMOID, // RELU
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }


	void activationBackward(int n, int c, int h, int w, value_type* srcData, value_type* dstData, value_type *srcDiffData, value_type **dstDiffData)
    {
        resize(n*c*h*w, dstDiffData);
        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                tensorFormat,
                                                dataType,
                                                n, c,
                                                h,
                                                w) );
		checkCUDNN( cudnnSetTensor4dDescriptor(srcDiffTensorDesc,
												tensorFormat,
												dataType,
												n, c,
												h,
												w) );
		checkCUDNN( cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
												tensorFormat,
												dataType,
												n, c,
												h,
												w) );
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnActivationBackward(cudnnHandle,
                                            CUDNN_ACTIVATION_SIGMOID, // RELU
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
											srcDiffTensorDesc,
											srcDiffData,
                                            dstTensorDesc,
                                            dstData,
                                            &beta,
											dstDiffTensorDesc,
											*dstDiffData
											) );    
    }


    int classify_example(const char* fname, const Layer_t& conv1,
                          const Layer_t& conv2,
                          const Layer_t& ip1,
                          const Layer_t& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        std::string sFilename(fname);
        std::cout << "Loading image " << sFilename << std::endl;
        // load gray-scale image from disk
        try
        {
            npp::loadImage(sFilename, oHostSrc);
        }
        catch (npp::Exception &rException)
        {
            FatalError(rException.toString());
        }
        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < IMAGE_H; i++)
        {
            for (int j = 0; j < IMAGE_W; j++)
            {   
                int idx = IMAGE_W*i + j;
                imgData_h[idx] = *(oHostSrc.data() + idx) / value_type(255);
            }
        }

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        convoluteForward(conv1, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        convoluteForward(conv2, n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(ip2, n, c, h, w, srcData, &dstData);
        softmaxForward(n, c, h, w, dstData, &srcData);

        const int max_digits = 10;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, srcData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (result[id] < result[i]) id = i;
        }

        if (DEBUG) printDeviceVector("Resulting weights from Softmax: \n",n*c*h*w, srcData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }

	int predictExample(const value_type** image_data, value_type target, const Layer_t& input, const Layer_t& hidden){
		
		value_type *image_data_d = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < N; i++)
        {
			imgData_h[i] = (*image_data)[i] / value_type(255);
			#if 0
			std::cout<<(imgData_h[i]>0?"#":" ")<<" ";
			if (i%IMAGE_W==IMAGE_W-1)
				std::cout<<std::endl;
			#endif
        }
		resize(IMAGE_H*IMAGE_W, &image_data_d);
		checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyHostToDevice) );
		int id = predictExampleDevice(image_data_d, target, input, hidden);
        checkCudaErrors( cudaFree(image_data_d) );
		return id;
	}


	int predictExampleDevice(const value_type* image_data_d, value_type target, const Layer_t& input, const Layer_t& hidden){
		int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
		if (DEBUG) std::cout<<std::endl<<"Target: "<<target<<std::endl;

		// Setup Variables for forward propagation
        //checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) ); 
		resize(IMAGE_H*IMAGE_W, &srcData);
		checkCudaErrors( cudaMemcpy(srcData, image_data_d,  IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        n = c = 1; h = IMAGE_H; w = IMAGE_W;
		// Perform Forward propagation
       	if (DEBUG) std::cout << "Performing forward propagation ...\n";
 
        if (DEBUG) printDeviceVector("input: \n", input.inputs, srcData);
        fullyConnectedForward(input, n, c, h, w, srcData, &dstData);
        if (DEBUG) printDeviceVector("fullyConnectedforward: \n", input.outputs, dstData);
        activationForward(n, c, h, w, dstData, &srcData);
		checkCudaErrors( cudaMemcpy(input.output_d, srcData, input.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        if (DEBUG) printDeviceVector("Hidden layer outputs: \n", n*c*h*w, input.output_d);

        fullyConnectedForward(hidden, n, c, h, w, srcData, &dstData);
        if (DEBUG) printDeviceVector("fullyConnectedforward: \n", hidden.outputs, dstData);
        activationForward(n, c, h, w, dstData, &srcData);
		checkCudaErrors( cudaMemcpy(hidden.output_d, srcData, hidden.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		if (DEBUG) printDeviceVector("Output layer outputs: \n", n*c*h*w, hidden.output_d);
		
		// Setup Variables for backward propagation
        const int max_digits = hidden.outputs; //n*c*h*w; //10;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, srcData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++){
            if (result[id] < result[i]) 
				id = i;
        }

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
		return id;
	}

	int learnExample(const value_type** image_data, value_type target, const Layer_t& input,
                          const Layer_t& hidden)
    {
       
		value_type *image_data_d = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < N; i++)
        {
			imgData_h[i] = (*image_data)[i] / value_type(255);
			#if 0
			std::cout<<(imgData_h[i]>0?"#":" ")<<" ";
			if (i%IMAGE_W==IMAGE_W-1)
				std::cout<<std::endl;
			#endif
        }
		resize(IMAGE_H*IMAGE_W, &image_data_d);
		checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyHostToDevice) );
		int id = predictExampleDevice(image_data_d, target, input, hidden);
		if (DEBUG) std::cout << "Prediction: "<<id << std::endl;
		
		
		value_type *srcDiffData = NULL, *dstDiffData = NULL, *targetData = NULL;
		int n, c, h, w;

		// Perform backward propagation
		if (DEBUG) std::cout<<"\nPerforming backward propagation ...\n";
		c = hidden.outputs; n = h = w = 1;

		getBackPropData(hidden, hidden, target, dstDiffData, &targetData, &srcDiffData, true);
		//THEORY: delW2 = (target-output)*output*(1-output)
		activationBackward(n, c, h, w, hidden.output_d, targetData, srcDiffData, &dstDiffData);
		checkCudaErrors( cudaMemcpy(hidden.del_d, dstDiffData, hidden.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		if (DEBUG) printDeviceVector("delW2: \n", hidden.outputs, hidden.del_d);

		c = input.outputs;
		getBackPropData(input, hidden, target, dstDiffData, &targetData, &srcDiffData, false);
		//THEORY: del_W1 = (del_W2*W2')*hidden_output*(1-hidden_output)
		if (DEBUG) printDeviceVector("\thidden_output: \n", input.outputs, input.output_d);
		activationBackward(n, c, h, w, input.output_d, targetData, srcDiffData, &dstDiffData); 
		checkCudaErrors( cudaMemcpy(input.del_d, dstDiffData, input.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		if (DEBUG) printDeviceVector("delW1: \n", input.outputs, input.del_d);


		fullyConnectedBackward(hidden, input.output_d);
		fullyConnectedBackward(input, image_data_d);

		// Free Memory
        checkCudaErrors( cudaFree(image_data_d) );
        checkCudaErrors( cudaFree(srcDiffData) );
        checkCudaErrors( cudaFree(dstDiffData) );
		checkCudaErrors( cudaFree(targetData) );
        return id;
    }
	
	void getBackPropData(const Layer_t& layer, const Layer_t& next_layer, value_type target, value_type* dstDiffData, 
			value_type** targetData, value_type** srcDiffData, 
			bool last_layer)
	{
		
		const int max_digits = layer.outputs;
        value_type srcData[max_digits];
        checkCudaErrors( cudaMemcpy(srcData, layer.output_d, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
		resize(max_digits, srcDiffData);
		resize(max_digits, targetData);
		if (last_layer){

			value_type srcDiffData_h[max_digits];
			value_type targetData_h[max_digits];

			for (int i = 0; i<max_digits; i++){
				targetData_h[i] = i==target?1:0;
				srcDiffData_h[i] = targetData_h[i]-srcData[i];
			}

			//checkCudaErrors( cudaMalloc(&srcDiffData, max_digits*sizeof(value_type)) );
			checkCudaErrors( cudaMemcpy(*srcDiffData, srcDiffData_h, max_digits*sizeof(value_type),cudaMemcpyHostToDevice) );
			//checkCudaErrors( cudaMalloc(&targetData, max_digits*sizeof(value_type)) );
			checkCudaErrors( cudaMemcpy(*targetData, targetData_h, max_digits*sizeof(value_type),cudaMemcpyHostToDevice) );
		
		}else{
			//THEORY: del_W1 = (del_W2*W2')*hidden_output*(1-hidden_output)
        	value_type alpha = value_type(1), beta = value_type(0);
			if (DEBUG) printDeviceVector("\tW2: \n", next_layer.inputs*next_layer.outputs, next_layer.data_d);
			checkCudaErrors( cublasDgemv(cublasHandle, CUBLAS_OP_N,
                                      next_layer.inputs, next_layer.outputs,
                                      &alpha,
                                      next_layer.data_d, next_layer.inputs,
                                      dstDiffData, 1,
                                      &beta,
                                      *srcDiffData, 1) );
			if (DEBUG) printDeviceVector("\tsrcDiffData: \n", next_layer.inputs, *srcDiffData);
		}
	}

	static void loadData(value_type **training_data, value_type **testing_data,
		 value_type **training_target, value_type **testing_target,
		 int &total_train_size, int &total_test_size){
		std::string name;
		total_train_size = 0;
		total_test_size = 0;
		std::string fname;
		std::stringstream error_s;
		// Calculate total training and testing size
		for (int t=0; t<2; t++){
			name = t==0?"train":"test";
			for (int d=0; d<10; d++){
				std::stringstream sstm;
				sstm<<"data/"<<name<<d<<".bin";
				fname = sstm.str();
				//std::cout<<"Calculating file "<<fname<<"\t";
				std::ifstream dataFile (fname.c_str(), std::ios::in | std::ios::binary);
				if (!dataFile)
				{
					error_s << "Error opening file " << fname; 
					FatalError(error_s.str());
				}

				dataFile.seekg(0, std::ios::end);
				size_t file_size = static_cast<std::string::size_type>(dataFile.tellg());
				dataFile.seekg(0, std::ios::beg);		
				dataFile.close();
				//std::cout<<file_size<<std::endl;
				if (t==0)
					total_train_size += file_size;
				else
					total_test_size += file_size;
			 }
		}

		*training_data = new value_type[total_train_size];
		*testing_data = new value_type[total_test_size];
		*training_target = new value_type[total_train_size/N];
		*testing_target = new value_type[total_test_size/N];
		total_train_size = 0;
		total_test_size = 0;
		for (int t=0; t<2; t++){
			name = t==0?"train":"test";
			for (int d=0; d<10; d++){
				std::stringstream sstm;
				sstm<<"data/"<<name<<d<<".bin";
				fname = sstm.str();
				//std::cout<<"Reading file "<<fname;
				std::ifstream dataFile (fname.c_str(), std::ios::in | std::ios::binary);
				if (!dataFile)
				{
					error_s << "Error opening file " << fname; 
					FatalError(error_s.str());
				}

				dataFile.seekg(0, std::ios::end);
				size_t file_size = static_cast<std::string::size_type>(dataFile.tellg());
				dataFile.seekg(0, std::ios::beg);		
				
				char *data = new char[file_size];
				if (!dataFile.read (data, file_size)) 
				{
					error_s << "Error reading file " << fname; 
					FatalError(error_s.str());
				}
				dataFile.close();

				value_type v;
				int m = file_size/N;
				//std::cout<<" "<<file_size<<" "<<m<<"\n";
				for (int i=0; i<file_size; i++){
					v = static_cast<value_type>((uint8_t)data[(i/N)+m*(i%N) ]);
					if (t==0){
						(*training_data)[total_train_size+i] = v;
						if (i<m)
							(*training_target)[total_train_size/N+i] = d;
					}
					else {  
						(*testing_data)[total_test_size+i] = v;
						if (i<m)
							(*testing_target)[total_test_size/N+i] = d;
					}
				}
				if (t==0)
					total_train_size += file_size;
				else
					total_test_size += file_size;
				delete [] data; 
			 }
		}
	}
};

inline bool IsAppBuiltAs64()
{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}

bool loadWeights(const char* filename, size_t size, value_type* matrix){
    std::ifstream myfile(filename, std::ios::in | std::ios::binary);
    if (myfile.is_open()){
        myfile.read((char*)matrix, size*sizeof(value_type));
    }else{
        std::cout<<"Error reading file "<<filename<<std::endl;
        return false;
    }
}

bool saveWeights(const char* filename, size_t size, value_type* matrix){
    std::ofstream myfile(filename, std::ios::out | std::ios::binary);
    if (myfile.is_open()){
        myfile.write((char*)matrix, size*sizeof(value_type));
    }else{
        std::cout<<"Error saving file "<<filename<<std::endl;
        return false;
    }
}


int main(int argc, char **argv){	
	if(sizeof(void*) != 8)
    {
#ifndef __aarch32__
      std::cout <<"With the exception of ARM, " << argv[0] << " is only supported on 64-bit OS and the application must be built as a 64-bit target. Test is being waived.\n";
      exit(EXIT_WAIVED);
#endif
    }

	srand(time(NULL));
	srand(rand());srand(rand());

	// Define and initialize network
	network_t mnist;
	Layer_t input(N,100);
	Layer_t hidden(100,10);
	
	value_type *train_data, *testing_data;
	value_type *train_target, *testing_target;

	// Read training data
	value_type *training_data;
	value_type *training_target;
	int total_train_data, total_test_data;
	mnist.loadData(&training_data, &testing_data, &training_target, &testing_target, total_train_data, total_test_data);
	std::cout<<"\n\nData Loaded. Training examples:"<<total_train_data/N<<" Testing examples:"<<total_test_data/N<<std::endl;
	
	// Shuffle training data
	int m = total_train_data/N;
	int *perm = new int[m];
	for (int i=0; i<m; i++) perm[i] = i;
	std::random_shuffle(&perm[0],&perm[m]);

	// apply the permutation
	train_data = new value_type[m*N];
	train_target = new value_type[m];
	for (int i=0; i<m; i++){
		for (int j=0; j<N; j++){
			train_data[i*N+j] = training_data[perm[i]*N+j];
		}
		train_target[i] = training_target[perm[i]];
	}

	std::cout<<"Training Examples shuffled."<<std::endl;

	// Free some variables
	delete [] training_data;
	delete [] training_target;

	if (DEBUG) getchar();
    
    if (
	    loadWeights("input_data.bin", input.inputs*input.outputs, input.data_h) &&
	    loadWeights("input_bias.bin", input.outputs, input.bias_h) &&
	    loadWeights("hidden_data.bin", hidden.inputs*hidden.outputs, hidden.data_h) &&
	    loadWeights("hidden_bias.bin", hidden.outputs, hidden.bias_h)
    ){
 	
        input.copyDataToDevice();
        hidden.copyDataToDevice();
        std::cout<<"Weights from file loaded"<<std::endl;
    }else{
        std::cout<<"\n **** Learning started ****"<<std::endl;
        // Learn all examples till convergence
        int num_iterations = 5;
        while(num_iterations--){ // Use a better convergence criteria
            for (int i=0; i<m; i++){
                if (DEBUG) std::cout<<"\n\n\n\n\n";
                const value_type *training_example = train_data+i*N;
                value_type target = train_target[i];
                value_type predicted = mnist.learnExample(&training_example, target, input, hidden);
                if (DEBUG) getchar();
                else if (i%1000==0) std::cout<<"."<<std::flush;
                //std::cout<<"Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted<<"\n";
            }
        }
        std::cout<<"\n **** Learning completed ****\n";
	    
        input.copyDataToHost();
        hidden.copyDataToHost();
	    // Save the weights in a binary file
	    saveWeights("input_data.bin", input.inputs*input.outputs, input.data_h);
	    saveWeights("input_bias.bin", input.outputs, input.bias_h);
	    saveWeights("hidden_data.bin", hidden.inputs*hidden.outputs, hidden.data_h);
	    saveWeights("hidden_bias.bin", hidden.outputs, hidden.bias_h);
    }

	std::cout<<"\n **** Testing started ****"<<std::endl;
	// Read testing data
	int correct = 0;
	int n = total_test_data/N;
	for (int i=0; i<n; i++){
		const value_type *test_example = testing_data+i*N;
		value_type target = testing_target[i];
		value_type predicted = mnist.predictExample(&test_example, target, input, hidden);
		if (target == predicted){
			correct++;
		}
		if (!DEBUG && i%1000==0) std::cout<<"."<<std::flush;
		//std::cout<<"Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted<<"\n";
	}
	std::cout<<"\n **** Testing completed ****\n";

	std::cout<<"Correctly predicted "<<correct<<" examples out of "<<n<<std::endl;


    cudaDeviceReset();
	exit(EXIT_SUCCESS);
	return 0; 
}

