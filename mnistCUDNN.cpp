/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include <cublas_v2.h>
#include <cudnn.h>

#include "ImageIO.h"

#define value_type float

#define IMAGE_H 28
#define IMAGE_W 28
#define N (IMAGE_H*IMAGE_W)  // dimension of training data

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
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
}

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

void printHostVector(int size, value_type* vec){
    for (int i = 0; i < size; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl; 
}


void printDeviceVector(int size, value_type* vec_d)
{
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
	printHostVector(size, vec);
    delete [] vec;
}


struct Layer_t
{
    int inputs;
    int outputs;
    int kernel_dim; 	// linear dimension (i.e. size is kernel_dim * kernel_dim)
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;

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
		data_h = new value_type[size];
		bias_h = new value_type[size_b];
			
        checkCudaErrors( cudaMalloc(&data_d, size_ac) );
        checkCudaErrors( cudaMalloc(&bias_d, size_b_ac) );

        checkCudaErrors( cudaMemcpy(data_d, data_h, size_ac, cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(bias_d, bias_h, size_b_ac, cudaMemcpyHostToDevice) );

		/*
		printHostVector(size, data_h);
		std::cout<<"\n\n\n\n"<<std::endl;
		printHostVector(size_b, bias_h);
		*/
	};

    ~Layer_t()
    {
        delete [] data_h;
        checkCudaErrors( cudaFree(data_d) );
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
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
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
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCudaErrors( cublasDestroy(cublasHandle) );
    }
  public:
    network_t()
    {
        dataType = CUDNN_DATA_FLOAT;
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
        
        checkCudaErrors( cublasSgemv(cublasHandle, CUBLAS_OP_T,
                                      dim_x, dim_y,
                                      &alpha,
                                      ip.data_d, dim_x,
                                      srcData, 1,
                                      &beta,
                                      *dstData, 1) );

        h = 1; w = 1; c = dim_y;
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
                                            CUDNN_ACTIVATION_RELU,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
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

        std::cout << "Resulting weights from Softmax:" << std::endl;
        printDeviceVector(n*c*h*w, srcData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }

	int myforward(const value_type** image_data, value_type output, const Layer_t& input,
                          const Layer_t& hidden)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];

        // Plot to console and normalize image to be in range [0,1]
        for (int i = 0; i < N; i++)
        {
			imgData_h[i] = (*image_data)[i] / value_type(255);
			std::cout<<imgData_h[i]<<" ";
			if (i%IMAGE_W==0)
				std::cout<<std::endl;
        }
		std::cout<<std::endl<<output<<std::endl;

        std::cout << "Performing forward propagation ...\n";

        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );

        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        //convoluteForward(conv1, n, c, h, w, srcData, &dstData);
        //poolForward(n, c, h, w, dstData, &srcData);

        //convoluteForward(conv2, n, c, h, w, srcData, &dstData);
        //poolForward(n, c, h, w, dstData, &srcData);

        fullyConnectedForward(input, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);

		std::cout<<"input layer done.."<<std::endl;

        fullyConnectedForward(hidden, n, c, h, w, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);

        const int max_digits = 10;
        value_type result[max_digits];
        checkCudaErrors( cudaMemcpy(result, srcData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (result[id] < result[i]) id = i;
        }

        std::cout << "Resulting weights from propogation:" << std::endl;
        printDeviceVector(n*c*h*w, srcData);

        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
        return id;
    }

	static void loadData(value_type **training_data, value_type **testing_data,
		 value_type **training_output, value_type **testing_output,
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
		*training_output = new value_type[total_train_size/N];
		*testing_output = new value_type[total_test_size/N];
		total_train_size = 0;
		total_test_size = 0;
		for (int t=0; t<2; t++){
			name = t==0?"train":"test";
			for (int d=0; d<10; d++){
				std::stringstream sstm;
				sstm<<"data/"<<name<<d<<".bin";
				fname = sstm.str();
				std::cout<<"Reading file "<<fname;
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
				std::cout<<" "<<file_size<<" "<<m<<"\n";
				for (int i=0; i<file_size; i++){
					v = static_cast<value_type>((uint8_t)data[(i/N)+m*(i%N) ]);
					if (t==0){
						(*training_data)[total_train_size+i] = v;
						if (i<m)
							(*training_output)[total_train_size/N+i] = d;
					}
					else {  
						(*testing_data)[total_test_size+i] = v;
						if (i<m)
							(*testing_output)[total_test_size/N+i] = d;
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

#define gete(array,i) (array+i*N)	

int main(int argc, char **argv){
	
	if(sizeof(void*) != 8)
    {
#ifndef __aarch32__
      std::cout <<"With the exception of ARM, " << argv[0] << " is only supported on 64-bit OS and the application must be built as a 64-bit target. Test is being waived.\n";
      exit(EXIT_WAIVED);
#endif
    }

	// Define and initialize network
	network_t mnist;
	Layer_t input(N,100);
	Layer_t hidden(100,10);
	
	value_type *training_data, *testing_data;
	value_type *training_output, *testing_output;


	// Read training data
	int total_train_data, total_test_data; // no of training data
	mnist.loadData(&training_data, &testing_data, &training_output, &testing_output, total_train_data, total_test_data);
	std::cout<<"Data Loaded "<<total_train_data<<" "<<total_test_data<<std::endl;
	//printHostVector(N, training_data+2*N);
/*
	//input.readBinFile("data/test0.bin", &data_h, &data_d, m, n);
	int k=0;
	while(true){
		value_type v = testing_data[k];
		std::cout<<k<<" "<<v<<std::endl;
		if (v!=0)
			break;

		k++;
	}
*/
	/*
	std::cout<<m<<" <- m"<<std::endl;	
	int s = 75;
	int s2 = 300;
	for (int i=s+0; i<s+10; i++){
		for (int j=s2+0; j<s2+10; j++){
			std::cout<<(((unsigned)data_h[j*m+i])&0x000000ff)<<"  ";
		}
		std::cout<<std::endl;
	}*/


	
	// shuffle training data
	int m = total_train_data/N;
	int *perm = new int[m];
	for (int i=0; i<m; i++)
		perm[i] = i;

	std::random_shuffle(&perm[0],&perm[m]);
	//std::cout<<"\n\n2:"<<perm[2]<<std::endl;
	//printHostVector(N, training_data+perm[2]*N);

	// apply the permutation
	value_type *train_data = new value_type[total_train_data];
	value_type *train_output = new value_type[m];
	for (int i=0; i<m; i++){
		for (int j=0; j<N; j++){
			train_data[i*N+j] = training_data[perm[i]*N+j];
		}
		train_output[i] = training_output[perm[i]];
	}
	delete [] training_data;
	delete [] training_output;	
	// Learn all examples till convergence
	int num_iterations = 1;
	while(num_iterations--){ // Use a better convergence criteria
		for (int i=0; i<m; i++){
			const value_type *training_example = gete(train_data,i);
			value_type output = train_output[i];
			mnist.myforward(&training_example,output,input,hidden);
			break;	
			//Feedforward input layer;
			//Feedforward hidden layer;
			// Get output

			// Calculate deltas
			// FeedBackward
			// Feedbackward
		}
	}
	
	
	// Save the weights in a binary file

	// Read testing data
	

	exit(EXIT_SUCCESS);
	return 0; 
}

int main2(int argc, char *argv[])
{
    if (argc > 2)
    {
        std::cout << "Test usage:\nmnistCUDNN [image]\nExiting...\n";
        exit(EXIT_FAILURE);
    }
    
    if(sizeof(void*) != 8)
    {
#ifndef __aarch32__
      std::cout <<"With the exception of ARM, " << argv[0] << " is only supported on 64-bit OS and the application must be built as a 64-bit target. Test is being waived.\n";
      exit(EXIT_WAIVED);
#endif
    }

    std::string image_path;
    network_t mnist;

    Layer_t conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
    Layer_t conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
    Layer_t   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
    Layer_t   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
    
    if (argc == 1)
    {
        int i1,i2,i3;
        get_path(image_path, first_image, argv[0]);
        i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
        get_path(image_path, second_image, argv[0]);
        i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
        
        get_path(image_path, third_image, argv[0]);
        i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

        std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
        if (i1 != 1 || i2 != 3 || i3 != 5)
        {
            std::cout << "\nTest failed!\n";
            FatalError("Prediction mismatch");
        }
        else
        {
            std::cout << "\nTest passed!\n";
        }
    }
    else
    {
        int i1 = mnist.classify_example(argv[1], conv1, conv2, ip1, ip2);
        std::cout << "\nResult of classification: " << i1 << std::endl;
    }
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
