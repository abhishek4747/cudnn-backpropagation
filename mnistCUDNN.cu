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

 * The sample can work in single, double, but it
 * assumes the data in files is stored in single precision
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ctime>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>

#include "ImageIO.h"
#include "error_util.h"

/******************************************************************************
 * MACROS
 *****************************************************************************/

// #define MATRIX_DATA_TYPE_FLOAT
#define MATRIX_DATA_TYPE_DOUBLE

#ifdef MATRIX_DATA_TYPE_FLOAT
#define MATRIX_DATA_TYPE float
#else
#ifdef MATRIX_DATA_TYPE_DOUBLE
#define MATRIX_DATA_TYPE double
#endif
#endif

#ifndef MATRIX_DATA_TYPE
#error "MATRIX_DATA_TYPE not defined"
#endif

#if defined(MATRIX_DATA_TYPE_FLOAT)
	#define CUBLAS_GEMM cublasSgemm
 	#define CUBLAS_GEAM cublasSgeam
 	#define CUBLAS_GEMV cublasSgemv
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
	#define CUBLAS_GEMM cublasDgemm
 	#define CUBLAS_GEAM cublasDgeam
 	#define CUBLAS_GEMV cublasDgemv
#endif

#define MSIZE(a) ((a)*sizeof(value_type))

#define IMAGE_H (28)
#define IMAGE_W (28)
#define N (IMAGE_H*IMAGE_W)  // dimension of training data

#define minn(a,b) (a<b?a:b)
#define maxx(a,b) (a>b?a:b)
#define minnn(a,b,c) (minn(minn(a,b),c))
#define maxxx(a,b,c) (maxx(maxx(a,b),c))

//#define print(a) (std::cout<<std::setprecision(0)<<std::fixed<<a)
#define print(a) (std::cout<<std::fixed<<a)
#define println(a) (print(a<<std::endl<<std::flush))

#define DEBUG (0)
//#define VERBOSE (0)

#ifdef VERBOSE
	#define vprint(a) print(a)
	#define vprintln(a) println(a)
#else
	#define vprint(a)
	#define vprintln(a)
#endif

#ifdef DEBUG
	#define dprint(a) print(a)
	#define dprintln(a) println(a)
#else
	#define dprint(a)
	#define dprintln(a)
#endif

#define EXIT_WAIVED 0

/******************************************************************************
 * CONSTANTS
 *****************************************************************************/

const std::string weights_folder = "bins/";
double learning_rate = 0.1;

/******************************************************************************
 * HELPER FUNCTIONS for classes
 *****************************************************************************/

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
	sFilename = (std::string("datav5/") + std::string(fname));
}

template <typename value_type> 
void printHostVector(std::string str, int size, value_type* vec){
	println(str<<" ("<<size<<") ");
	for (int i = 0; i < minn(size,400); i++)
	{
		print(vec[i] << " ");
	}
	println(" "); 
}

template <typename value_type>
void printDeviceVector(std::string str, int size, value_type* vec_d)
{
	value_type *vec;
	vec = new value_type[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, MSIZE(size), cudaMemcpyDeviceToHost);
	printHostVector(str, size, vec);
	delete [] vec;
}

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
	std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
	std::stringstream error_s;
	if (!dataFile)
	{
		error_s << "Error opening file " << fname; 
		FatalError(error_s.str());
	}
	// we assume the data stored is always in float precision
	float* data_tmp = new float[size];
	int size_b = size*sizeof(float);
	if (!dataFile.read ((char*) data_tmp, size_b)) 
	{
		error_s << "Error reading file " << fname; 
		FatalError(error_s.str());
	}

	for (int i = 0; i < size; i++)
	{
		data_h[i] = value_type(data_tmp[i]);
	}
	delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
	*data_h = new value_type[size];

	readBinaryFile<value_type>(fname, size, *data_h);

	int size_b = MSIZE(size);
	checkCudaErrors( cudaMalloc(data_d, size_b) );
	checkCudaErrors( cudaMemcpy(*data_d, *data_h,
								size_b,
								cudaMemcpyHostToDevice) );
}

template <class value_type>
void readImage(const char* fname, value_type* imgData_h)
{
	// declare a host image object for an 8-bit grayscale image
	npp::ImageCPU_8u_C1 oHostSrc;
	std::string sFilename(fname);
	println("Loading image " << sFilename);
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
			imgData_h[idx] = value_type(*(oHostSrc.data() + idx) / double(255));
		}
	} 
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
	value_type *vec;
	vec = new value_type[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, MSIZE(size), cudaMemcpyDeviceToHost);
	std::cout.precision(5);
	std::cout.setf( std::ios::fixed, std::ios::floatfield );
	for (int i = 0; i < size; i++)
	{
		print(value_type(vec[i]) << " ");
	}
	println(" ");
	delete [] vec;
}

/******************************************************************************
 * demonstrate different ways of setting tensor descriptor
 *****************************************************************************/

//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
					cudnnTensorFormat_t& tensorFormat,
					cudnnDataType_t& dataType,
					int n,
					int c,
					int h,
					int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
	checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
											tensorFormat,
											dataType,
											n, c,
											h,
											w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
	const int nDims = 4;
	int dimA[nDims] = {n,c,h,w};
	int strideA[nDims] = {c*h*w, h*w, w, 1};
	checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
											dataType,
											4,
											dimA,
											strideA ) ); 
#else
	checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
											dataType,
											n, c,
											h, w,
											c*h*w, h*w, w, 1) );
#endif
}

/******************************************************************************
 * Defining Layer Types
 *****************************************************************************/

typedef enum {
		CONV_LAYER	= 0,
		POOL_LAYER	= 1,
		FC_LAYER	= 2,
		ACT_LAYER	= 3,
		NORM_LAYER	= 4,
		SOFTMAX_LAYER= 5
} LayerType;

/******************************************************************************
 * Layer_t struct : contains information about layers
 *****************************************************************************/
template <class value_type>
struct Layer_t
{
	LayerType layerType;
	std::string layername;
	int n;

	int inputs, outputs, kernel_dim; // linear dimension (i.e. size is kernel_dim * kernel_dim)
	int  w_size, b_size, d_size;

	int in_height, in_width;
	int out_height, out_width;

	value_type *data_h, 	*data_d;
	value_type *bias_h, 	*bias_d;

	value_type *output_d,	*del_d;

	// Convolutional Layer
	cudnnConvolutionDescriptor_t convDesc;
	cudnnTensorDescriptor_t convBiasTensorDesc;
	cudnnFilterDescriptor_t convFilterDesc;
	cudnnTensorDescriptor_t convSrcTensorDesc, convDstTensorDesc;
	cudnnConvolutionFwdAlgo_t convFwdAlgo;
	cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
	size_t convFwdSizeInBytes, convBwdDataSizeInBytes, convBwdFilterSizeInBytes;

	// Pooling Layer
	cudnnPoolingDescriptor_t poolDesc;
	cudnnTensorDescriptor_t poolSrcTensorDesc, poolDstTensorDesc;
	cudnnFilterDescriptor_t poolFilterDesc;
	int size, stride;
	
	// Fully Connected Layer


	// Activation Layer
	cudnnActivationDescriptor_t  activDesc;
	cudnnTensorDescriptor_t actTensorDesc;


	// Normal Layer


	// Softmax Layer


	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;

	Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
				inputs(0), outputs(0), kernel_dim(0)
	{
		switch (sizeof(value_type))
		{
			case 4 : dataType = CUDNN_DATA_FLOAT; break;
			case 8 : dataType = CUDNN_DATA_DOUBLE; break;
			default : FatalError("Unsupported data type");
		}
		tensorFormat = CUDNN_TENSOR_NCHW;
		data_d = bias_d = output_d = del_d = NULL;
		n = 0;
		convFwdSizeInBytes = convBwdDataSizeInBytes = convBwdFilterSizeInBytes = 0;
	};

	~Layer_t()
	{
		if (data_h != NULL) 	delete [] data_h;
		if (bias_h != NULL) 	delete [] bias_h;

		if (data_d != NULL) 	checkCudaErrors( cudaFree(data_d) );
		if (bias_d != NULL) 	checkCudaErrors( cudaFree(bias_d) );
		if (output_d != NULL) 	checkCudaErrors( cudaFree(output_d) );
		if (del_d != NULL) 		checkCudaErrors( cudaFree(del_d) );

		if (layerType == CONV_LAYER){
			destroyConvLayer();
		} else if (layerType == POOL_LAYER){
			destroyPoolLayer();
		} else if (layerType == ACT_LAYER || layerType == SOFTMAX_LAYER || layerType == NORM_LAYER){
			destroyActLayer();
		} else if (layerType == FC_LAYER){
			destroyLayer();
		}
	}

	void setHandles(int _n)
	{
		if (_n==n)
			return;
		n  = _n;
		if (layerType==CONV_LAYER){
			createConvHandles();
		} else if (layerType==POOL_LAYER){
			createPoolHandles();
		} else if (layerType==ACT_LAYER || layerType==SOFTMAX_LAYER || layerType==NORM_LAYER){
			createActHandles();
		} else {	// FC_LAYER
			createFCHandles();
		}
	}

	void createPoolHandles(){
		int c, h, w;
		c = kernel_dim; h=in_height; w=in_width;
		setTensorDesc(poolSrcTensorDesc, tensorFormat, dataType, n, c, h, w);        

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims] = {n,c,h,w};
		checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolDesc,
													poolSrcTensorDesc,
													tensorDims,
													tensorOuputDimA) );
		n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

		out_height = h;
		out_width  = w;
		
		setTensorDesc(poolDstTensorDesc, tensorFormat, dataType, n, c, h, w);  

		b_size		= kernel_dim * out_width * out_height;
		outputs 	= b_size;
		inputs  	= kernel_dim * in_width * in_height; 
		
		if (output_d != NULL) 	checkCudaErrors( cudaFree(output_d) );
		if (del_d != NULL) 		checkCudaErrors( cudaFree(del_d) );

		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(n*outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(n*inputs)) );
	}

	void createFCHandles(){
		if (output_d != NULL) 	checkCudaErrors( cudaFree(output_d) );
		if (del_d != NULL) 		checkCudaErrors( cudaFree(del_d) );

		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(n*outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(n*inputs)) );
	}

	void createActHandles(){
		int c, h, w;
		h = w = 1; c = inputs;
		setTensorDesc(actTensorDesc, tensorFormat, dataType, n, c, h, w);

		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(n*outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(n*inputs)) );

	}

	void createConvHandles()
	{
		int c = inputs;
		int h = in_height;
		int w = in_width;

        checkCUDNN(cudnnSetTensor4dDescriptor(convSrcTensorDesc,
                                              tensorFormat,
                                              dataType,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetTensor4dDescriptor(convBiasTensorDesc,
                                              tensorFormat,
                                              dataType,
                                              1, outputs,
                                              1, 1));

        checkCUDNN(cudnnSetFilter4dDescriptor(convFilterDesc,
                                              dataType,
                                              tensorFormat,
                                              outputs,
                                              inputs, 
											  kernel_dim,
											  kernel_dim));
 
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,	//	padding
                                                   stride, stride,	//	stride
                                                   1, 1,	// 	upscaling
                                                   CUDNN_CROSS_CORRELATION));
        // Find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         convSrcTensorDesc,
                                                         convFilterDesc,
                                                         &n, &c, &h, &w));

        out_width 	= w;			
		out_height 	= h;


        checkCUDNN(cudnnSetTensor4dDescriptor(convDstTensorDesc,
                                              tensorFormat,
                                              dataType,
                                              n, c,
                                              h, w));
        cudnnHandle_t cudnnHandle;
		checkCUDNN( cudnnCreate(&cudnnHandle) );

		convFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        // checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
        //                                                convSrcTensorDesc,
        //                                                convFilterDesc,
        //                                                convDesc,
        //                                                convDstTensorDesc,
        //                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        //                                                0,
        //                                                &convFwdAlgo));
        
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           convSrcTensorDesc,
                                                           convFilterDesc,
                                                           convDesc,
                                                           convDstTensorDesc,
                                                           convFwdAlgo,
                                                           &convFwdSizeInBytes));
        
        convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  		// checkCUDNN( cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
  		// 													convFilterDesc,
  		// 													convDstTensorDesc,
  		// 													convDesc,
  		// 													convSrcTensorDesc,
  		// 													CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
  		// 													0,
  		// 													&convBwdDataAlgo));

  		checkCUDNN( cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
														convFilterDesc,
														convDstTensorDesc,
														convDesc,
														convSrcTensorDesc,
														convBwdDataAlgo,
														&convBwdDataSizeInBytes
														));

  		convBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  		// checkCUDNN( cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
  		// 														convSrcTensorDesc,
  		// 														convDstTensorDesc,
  		// 														convDesc,
  		// 														convFilterDesc,
  		// 														CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
  		// 														0,
  		// 														&convBwdFilterAlgo));
  		checkCUDNN( cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
												convSrcTensorDesc,
												convDstTensorDesc,
												convDesc,
												convFilterDesc,
												convBwdFilterAlgo,
												&convBwdFilterSizeInBytes));

  		println("handles: "<<(int)convFwdAlgo<<" "<<(int)convBwdDataAlgo<<" "<<(int)convBwdFilterAlgo);

        checkCUDNN( cudnnDestroy(cudnnHandle) );

        if (data_d != NULL) 	checkCudaErrors( cudaFree(data_d) );
		if (bias_d != NULL) 	checkCudaErrors( cudaFree(bias_d) );
		if (output_d != NULL) 	checkCudaErrors( cudaFree(output_d) );
		if (del_d != NULL) 		checkCudaErrors( cudaFree(del_d) );

        checkCudaErrors( cudaMalloc(&data_d, 	MSIZE(w_size)) );
		checkCudaErrors( cudaMalloc(&bias_d, 	MSIZE(b_size)) );
		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(n*outputs*out_height*out_width)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(n*d_size)) );
	}

	void initConvLayer(std::string _layername, int _inputs, int _outputs, int _kernel_dim, int _stride, int _in_height, int _in_width, int _d_size=0)
	{
		layerType 	= CONV_LAYER;
		layername 	= _layername;
		inputs 		= _inputs;
		outputs 	= _outputs;
		kernel_dim 	= _kernel_dim;
		stride 		= _stride;
		in_width 	= _in_width;
		in_height 	= _in_height;
		w_size 		= inputs*outputs*kernel_dim*kernel_dim;
		b_size 		= outputs;
		d_size 		= _d_size;

		data_h 	= new value_type[w_size];
		bias_h 	= new value_type[b_size];

		// Random Initialization
		// TODO : Fix this random initialization
		for (int i=0; i<w_size; i++)
			data_h[i] = (((value_type)rand())/(rand()+1))/100000;
		for (int i=0; i<b_size; i++)
			bias_h[i] = (((value_type)rand())/(rand()+1))/100000;
		
		
		checkCUDNN(cudnnCreateTensorDescriptor(&convSrcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&convDstTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&convBiasTensorDesc));

		setHandles(1);

		copyDataToDevice();
	}

	void initPoolLayer(std::string _layername, int _size, int _stride, const Layer_t<value_type>& conv)
	{
		layerType 	= POOL_LAYER;
		layername 	= _layername;
		size 		= _size;
		stride 		= _stride;
		w_size		= 0;
		kernel_dim  = conv.outputs;
		in_height 	= conv.out_height;
		in_width 	= conv.out_width;		

		checkCUDNN(cudnnCreateTensorDescriptor(&poolSrcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&poolDstTensorDesc));
		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
											   CUDNN_POOLING_MAX,
											   CUDNN_PROPAGATE_NAN,
											   size, size,
											   0, 0,
											   stride, stride));

		setHandles(1);
	}

	void initFCLayer(std::string _layername, int _inputs, int _outputs)
	{
		layerType 	= FC_LAYER;
		layername 	= _layername;
		inputs 		= _inputs;
		outputs 	= _outputs;
		kernel_dim 	= 1;
		w_size 		= inputs*outputs*kernel_dim*kernel_dim;
		b_size 		= outputs;

		data_h 	= new value_type[w_size];
		bias_h 	= new value_type[b_size];

		// Random Initialization
		// TODO : Fix this random initialization
		for (int i=0; i<w_size; i++)
			data_h[i] = (((value_type)rand())/(rand()+1))/100000;
		for (int i=0; i<b_size; i++)
			bias_h[i] = (((value_type)rand())/(rand()+1))/100000;			
		
		
		checkCudaErrors( cudaMalloc(&data_d, 	MSIZE(w_size)) );
		checkCudaErrors( cudaMalloc(&bias_d, 	MSIZE(b_size)) );
		
		setHandles(1);

		copyDataToDevice();
	}

	void initActLayer(std::string _layername, int _outputs){
		initLayer(_layername, ACT_LAYER, _outputs);
	}

	void initSoftmaxLayer(std::string _layername, int _outputs){
		initLayer(_layername, SOFTMAX_LAYER, _outputs);
	}

	

	void destroyConvLayer(){
		checkCUDNN(cudnnDestroyTensorDescriptor(convSrcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(convDstTensorDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(convFilterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(convBiasTensorDesc));
	}

	void destroyPoolLayer(){
		checkCUDNN(cudnnDestroyTensorDescriptor(poolSrcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(poolDstTensorDesc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
	}

	void destroyActLayer(){
		checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(actTensorDesc) );
	}

	void destroyLayer(){

	}

	void copyDataToDevice(){
		if (data_h!=NULL) 	checkCudaErrors( cudaMemcpy(data_d, 	data_h, 	MSIZE(w_size), 	cudaMemcpyHostToDevice) );
		if (bias_h!=NULL) 	checkCudaErrors( cudaMemcpy(bias_d, 	bias_h, 	MSIZE(b_size), 	cudaMemcpyHostToDevice) );
	}
	
	void copyDataToHost(){
		if (data_h!=NULL) 	checkCudaErrors( cudaMemcpy(data_h, 	data_d, 	MSIZE(w_size), 	cudaMemcpyDeviceToHost) );
		if (bias_h!=NULL) 	checkCudaErrors( cudaMemcpy(bias_h, 	bias_d, 	MSIZE(b_size), 	cudaMemcpyDeviceToHost) );
	}

	bool load(){
		std::string dtype = (sizeof(value_type)==4?"_float_":"_double_");
		return loadWeights(layername+dtype+"weights.bin", w_size, data_h) && loadWeights(layername+dtype+"bias.bin", b_size, bias_h);
	}

	bool save(){
		std::string dtype = (sizeof(value_type)==4?"_float_":"_double_");
		return saveWeights(layername+dtype+"weights.bin", w_size, data_h) && saveWeights(layername+dtype+"bias.bin", b_size, bias_h);
	}

	bool loadWeights(std::string filename, size_t size, value_type* matrix){
		filename = weights_folder+filename;
		std::ifstream myfile(filename.c_str(), std::ios::in | std::ios::binary);
		if (myfile.is_open()){
			myfile.read((char*)matrix, MSIZE(size));
			return true;
		}else{
			println("Error reading file "<<filename);
			return false;
		}
	}

	bool saveWeights(std::string filename, size_t size, value_type* matrix){
		filename = weights_folder+filename;
		std::ofstream myfile(filename.c_str(), std::ios::out | std::ios::binary);
		if (myfile.is_open()){
			myfile.write((char*)matrix, MSIZE(size));
			return true;
		}else{
			println("Error saving file "<<filename);
			return false;
		}
	}
private:
	void initLayer(std::string _layername, LayerType _layerType, int _outputs){
		layerType 	= _layerType;
		layername 	= _layername;
		inputs 		= _outputs;
		outputs 	= _outputs;
		kernel_dim 	= 1;
		w_size 		= 0;
		b_size 		= 0;
		
		checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
		checkCUDNN( cudnnCreateTensorDescriptor(&actTensorDesc) );
		checkCUDNN( cudnnSetActivationDescriptor(activDesc,
												CUDNN_ACTIVATION_RELU, //CUDNN_ACTIVATION_SIGMOID,
												CUDNN_PROPAGATE_NAN,
												0.0) );

		setHandles(1);
	}

	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
	}
};


/******************************************************************************
 * network_t class : contains all learning functions
 *****************************************************************************/

__global__ void getDiffDataD(int target, MATRIX_DATA_TYPE* diffData){
 	int idx = threadIdx.x;
 	if (idx==target)
 		diffData[idx] -= 1;
}

template <class value_type>
class network_t
{
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;
	value_type vOne, vZero;

	void createHandles()
	{
		checkCUDNN( cudnnCreate(&cudnnHandle) );
		checkCublasErrors( cublasCreate(&cublasHandle) );
	}

	void destroyHandles()
	{
		checkCUDNN( cudnnDestroy(cudnnHandle) );
		checkCublasErrors( cublasDestroy(cublasHandle) );
	}
  public:
	network_t()
	{
		vOne  = value_type(1);
		vZero = value_type(0);
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
		checkCudaErrors( cudaMalloc(data, MSIZE(size)) );
	}
	
	void addBias(const cudnnTensorDescriptor_t& convDstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
	{
		checkCUDNN( cudnnAddTensor( cudnnHandle, 
									&vOne, 
									layer.convBiasTensorDesc,
									layer.bias_d,
									&vOne,
									convDstTensorDesc,
									data) );
	}

	void fullyConnectedForward(const Layer_t<value_type>& layer,
						  int& n,
						  value_type* srcData)
	{
		if (n != 1)
		{
			FatalError("Not Implemented"); 
		}

		
		int dim_x = layer.inputs;
		int dim_y = layer.outputs;
		
		checkCudaErrors( cudaMemcpy(layer.output_d, layer.bias_d, MSIZE(dim_y), cudaMemcpyDeviceToDevice) );
		
		checkCublasErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_T,
                                  dim_x, dim_y,
                                  &vOne,
                                  layer.data_d, dim_x,
                                  srcData, 1,
                                  &vOne,
                                  layer.output_d, 1) );    

	}

	void convoluteForward(const Layer_t<value_type>& layer,
						  int& n, 
						  value_type* srcData)
	{

		if (DEBUG) printDeviceVector("Conv Weights:\n", layer.w_size, layer.data_d);
		if (DEBUG) printDeviceVector("Conv Bias:\n", layer.b_size, layer.bias_d);
		void* workSpace=NULL;
		if (layer.convFwdSizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,layer.convFwdSizeInBytes) );
		}
		checkCUDNN( cudnnConvolutionForward(cudnnHandle,
											  &vOne,
											  layer.convSrcTensorDesc,
											  srcData,
											  layer.convFilterDesc,
											  layer.data_d,
											  layer.convDesc,
											  layer.convFwdAlgo,
											  workSpace,
											  layer.convFwdSizeInBytes,
											  &vZero,
											  layer.convDstTensorDesc,
											  layer.output_d) );
		addBias(layer.convDstTensorDesc, layer, layer.outputs, layer.output_d);
		if (DEBUG) printDeviceVector("Conv Output:\n", layer.outputs*layer.out_height*layer.out_width, layer.output_d);
		if (layer.convFwdSizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}
	}

	void convoluteBackward(const Layer_t<value_type>& layer,
							int& n,
							value_type* diffData)
	{
		void* workSpace=NULL;

		
		if (layer.convBwdDataSizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,layer.convBwdDataSizeInBytes) );
		}
		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, 
												&vOne, 
												layer.convFilterDesc, layer.data_d, 
												layer.convDstTensorDesc, diffData, 
												layer.convDesc, layer.convBwdDataAlgo,
												workSpace, layer.convBwdDataSizeInBytes,
												&vZero, 
												layer.convSrcTensorDesc, layer.del_d));
		if (layer.convBwdDataSizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}
	}

	void poolForward(const Layer_t<value_type>& layer,
					  int& n, 
					  value_type* srcData)
	{


		if (DEBUG) printDeviceVector("Pooling Input:\n", layer.inputs, layer.output_d);
		checkCUDNN( cudnnPoolingForward(cudnnHandle,
										  layer.poolDesc,
										  &vOne,
										  layer.poolSrcTensorDesc,
										  srcData,
										  &vZero,
										  layer.poolDstTensorDesc,
										  layer.output_d) );
		if (DEBUG) printDeviceVector("Pooling Output:\n", layer.outputs, layer.output_d);
	}

	void poolBackward(const Layer_t<value_type>& layer,
						int& n,
						value_type* diffData, value_type* srcData)
	{

		if (DEBUG) printDeviceVector("Pooling back Input: ", layer.outputs, srcData);
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, 
											layer.poolDesc, 
											&vOne, 
											layer.poolDstTensorDesc, layer.output_d, 
											layer.poolDstTensorDesc, diffData,
											layer.poolSrcTensorDesc, srcData, 
											&vZero, 
											layer.poolSrcTensorDesc, layer.del_d));
		if (DEBUG) printDeviceVector("Pooling back Output: ", layer.inputs, layer.del_d);

	}

	void softmaxForward(const Layer_t<value_type>& layer, 
						int &n, value_type* srcData)
	{

		checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
										  CUDNN_SOFTMAX_ACCURATE ,
										  CUDNN_SOFTMAX_MODE_CHANNEL,
										  &vOne,
										  layer.actTensorDesc,
										  srcData,
										  &vZero,
										  layer.actTensorDesc,
										  layer.output_d) );
	}

	void getDiffData(const Layer_t<value_type>& layer, int target, value_type** diffData){
		resize(layer.outputs, diffData);
		value_type outputh[layer.outputs];
		checkCudaErrors( cudaMemcpy(outputh, layer.output_d, MSIZE(layer.outputs), cudaMemcpyDeviceToHost) );
		for (int i=0; i<layer.outputs; i++){
			if (i==target)
				outputh[i] -= 1 ;
		}
		checkCudaErrors( cudaMemcpy(*diffData, outputh, MSIZE(layer.outputs), cudaMemcpyHostToDevice) );
	}

	void softmaxBackward(const Layer_t<value_type>& layer, 
						int &n, 
						value_type* diffData, value_type* srcData)
	{
		checkCUDNN( cudnnSoftmaxBackward(cudnnHandle,
										  CUDNN_SOFTMAX_ACCURATE ,
										  CUDNN_SOFTMAX_MODE_CHANNEL,
										  &vOne,
										  layer.actTensorDesc,
										  layer.output_d,
										  layer.actTensorDesc,
										  diffData,
										  &vZero,
										  layer.actTensorDesc,
										  layer.del_d) );
	}

	void activationForward(const Layer_t<value_type>& layer, 
							int &n, value_type* srcData)
	{
		checkCUDNN( cudnnActivationForward(cudnnHandle,
											layer.activDesc,
											&vOne,
											layer.actTensorDesc,
											srcData,
											&vZero,
											layer.actTensorDesc,
											layer.output_d) );    
	}

	void fullyConnectedBackward(const Layer_t<value_type>& layer,
								int &n, value_type* srcData)
	{
		checkCudaErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_N,
									  layer.inputs, layer.outputs,
									  &vOne,
									  layer.data_d, layer.inputs,
									  srcData, 1,
									  &vZero,
									  layer.del_d, 1) );
	}

	void activationBackward(const Layer_t<value_type>& layer,
							int &n, 
							value_type *srcDiffData, value_type* srcData)
	{
		checkCUDNN( cudnnActivationBackward(cudnnHandle,
											layer.activDesc,
											&vOne,
											layer.actTensorDesc,
											layer.output_d,
											layer.actTensorDesc,
											srcDiffData,
											layer.actTensorDesc,
											srcData,
											&vZero,
											layer.actTensorDesc,
											layer.del_d
											) );    
	}

	void fullyConnectedUpdateWeights(const Layer_t<value_type>& layer, value_type* diffData, value_type* srcData){
		int dim_x = layer.inputs;
		int dim_y = layer.outputs;
		int dim_z = 1;
		value_type* dstData = NULL;
		resize(dim_x*dim_y, &dstData);

		// checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, MSIZE(ip.outputs), cudaMemcpyDeviceToDevice) );
		//if (DEBUG) printDeviceVector("last_input: \n", layer.inputs, last_input);
		//if (DEBUG) printDeviceVector("del_W: \n", layer.outputs, layer.del_d);
		
		checkCudaErrors( CUBLAS_GEMM(cublasHandle, 
									  CUBLAS_OP_N, CUBLAS_OP_N,
									  dim_x, dim_y, dim_z,
									  &vOne,
									  srcData, dim_x,
									  diffData, dim_z,
									  &vZero,
									  dstData, dim_x) );
		
		// if (DEBUG) printDeviceVector("\tdelta_W (del_W*hidden_input): \n", layer.inputs*layer.outputs, dstData);

		value_type lr = value_type(-learning_rate); // learning rate
		//checkCudaErrors( cublasDscal(cublasHandle, ip.inputs*ip.outputs, &alpha, ip.data_d, 1); 
		const value_type* B = layer.data_d;
		// C = α op ( A ) + β * C
		// C = 0.1 * delta_W2 + C
		// if (DEBUG) printDeviceVector("\tW = W + 0.1*delta_W: old\n", dim_x*dim_y, layer.data_d);
		
		checkCudaErrors( CUBLAS_GEAM(cublasHandle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										dim_x, dim_y,
										&lr,
										dstData, dim_x,
										&vOne,
										B, dim_x,
										layer.data_d, dim_x) );
		// if (DEBUG) printDeviceVector("\tW: \n", dim_x*dim_y, layer.data_d);

		// place bias into dstData
		dim_x = 1;
		const value_type* B2 = layer.bias_d;
		// if (DEBUG) printDeviceVector("\tdel_W:\n", layer.outputs, layer.del_d);
		// if (DEBUG) printDeviceVector("\tB = B + 0.1*del_W: old\n", layer.outputs, layer.bias_d);
		checkCudaErrors( CUBLAS_GEAM(cublasHandle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										dim_x, dim_y,
										&lr,
										diffData, dim_x,
										&vOne,
										B2, dim_x,
										layer.bias_d, dim_x) );
		// if (DEBUG) printDeviceVector("\tB:\n", layer.outputs, layer.bias_d);

		checkCudaErrors( cudaFree(dstData));
	}

	void convolutionalUpdateWeights(const Layer_t<value_type>& layer, value_type* diffData, value_type* srcData)
	{

		if (DEBUG) println("Convolutional Update Weights:");

		value_type *gconvB = NULL, *gconvW = NULL;
		resize(layer.outputs, &gconvB);
		resize(layer.w_size, &gconvW);
		
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, 
												&vOne, 
												layer.convDstTensorDesc, diffData, 
												&vZero, 
												layer.convBiasTensorDesc, gconvB));

		if (DEBUG) printDeviceVector(" gconvB: ", layer.outputs, gconvB);

		void* workSpace=NULL;
		
		if (layer.convBwdFilterSizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,layer.convBwdFilterSizeInBytes) );
		}
		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, 
												&vOne, 
												layer.convSrcTensorDesc, srcData, 
												layer.convDstTensorDesc, diffData, 
												layer.convDesc, layer.convBwdFilterAlgo,
												workSpace, layer.convBwdFilterSizeInBytes,
												&vZero, 
												layer.convFilterDesc, gconvW));
		if (layer.convBwdFilterSizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}

		if (DEBUG) printDeviceVector(" gconvW: ", layer.w_size, gconvW);

		value_type lr = value_type(-learning_rate); // learning rate
		checkCudaErrors(cublasDaxpy(cublasHandle, 
									layer.outputs*layer.inputs*layer.kernel_dim*layer.kernel_dim,
									&lr, 
									gconvW, 1, 
									layer.data_d, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, 
									layer.outputs,
									&lr, 
									gconvB, 1, 
									layer.bias_d, 1));

		if (DEBUG) printDeviceVector(" Updated Weights: ", layer.w_size, layer.data_d);
		if (DEBUG) printDeviceVector(" Updated Bias: ", layer.b_size, layer.bias_d);
		
		checkCudaErrors( cudaFree(gconvB) );
		checkCudaErrors( cudaFree(gconvW) );
		if (DEBUG) getchar();
	}

	int predict_example(value_type* image_data_d, 
						const Layer_t<value_type>& conv1,
						const Layer_t<value_type>& pool1,
						const Layer_t<value_type>& conv2,
						const Layer_t<value_type>& pool2,
						const Layer_t<value_type>& fc1,
						const Layer_t<value_type>& fc1act,
						const Layer_t<value_type>& fc2,
						const Layer_t<value_type>& fc2act)
	{
		int n;
		// if (DEBUG) println("Performing forward propagation ...");

		n = 1;

		convoluteForward(conv1, n, image_data_d);
		poolForward(pool1, 		n, conv1.output_d);

		convoluteForward(conv2, n, pool1.output_d);
		poolForward(pool2, 		n, conv2.output_d);

		fullyConnectedForward(fc1, 	n, pool2.output_d);
		activationForward(fc1act, 	n, fc1.output_d);
		
		// lrnForward(n, srcData, &dstData);

		fullyConnectedForward(fc2, 	n, fc1act.output_d);
		activationForward(fc2act, 	n, fc2.output_d);
		// softmaxForward(fc2act, 	n, fc2.output_d);

		const int max_digits = fc2act.outputs;
		
		value_type result[max_digits];
		checkCudaErrors( cudaMemcpy(result, fc2act.output_d, MSIZE(max_digits), cudaMemcpyDeviceToHost) );
		int id = 0;
		for (int i = 1; i < max_digits; i++)
		{
			if ((result[id]) < (result[i])) id = i;
		}

		return id;
	}
	/*
	int predict_example(value_type* image_data_d,
						const Layer_t<value_type>& fc1,
						const Layer_t<value_type>& fc1act,
						const Layer_t<value_type>& fc2,
						const Layer_t<value_type>& fc2act)
	{
		int n, c, h, w;
		// if (DEBUG) println("Performing forward propagation ...");

		n = c = 1; h = IMAGE_H; w = IMAGE_W;

		fullyConnectedForward(fc1, 	n, c, h, w, image_data_d);
		activationForward(fc1act, 	n, c, h, w, fc1.output_d);
		
		// lrnForward(n, c, h, w, srcData, &dstData);

		fullyConnectedForward(fc2, 	n, c, h, w, fc1act.output_d);
		// activationForward(fc2act, 	n, c, h, w, fc2.output_d);
		softmaxForward(fc2act, 	n, c, h, w, fc2.output_d);

		const int max_digits = fc2act.outputs;
		
		value_type result[max_digits];
		checkCudaErrors( cudaMemcpy(result, fc2act.output_d, MSIZE(max_digits), cudaMemcpyDeviceToHost) );
		int id = 0;
		for (int i = 1; i < max_digits; i++)
		{
			if ((result[id]) < (result[i])) id = i;
		}

		return id;
	}
	*/

	int learn_example(value_type* image_data_d, 
						const Layer_t<value_type>& conv1,
						const Layer_t<value_type>& pool1,
						const Layer_t<value_type>& conv2,
						const Layer_t<value_type>& pool2,
						const Layer_t<value_type>& fc1,
						const Layer_t<value_type>& fc1act,
						const Layer_t<value_type>& fc2,
						const Layer_t<value_type>& fc2act,
						int target)
	{
		int n,c;
		
		int id = predict_example(image_data_d, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);

		//if (DEBUG) println("Performing backward propagation ...");
		n = 1; c = fc2act.outputs;

		value_type *diffData = NULL;
		resize(c, &diffData);
		checkCudaErrors( cudaMemcpy(diffData, fc2act.output_d, MSIZE(c), cudaMemcpyDeviceToDevice) );
		// getDiffData(fc2act, target, &diffData);
		getDiffDataD<<<1, c>>>(target, diffData);
		cudaDeviceSynchronize();

		activationBackward(fc2act,	n, diffData, fc2.output_d);
		// softmaxBackward(fc2act,		n, diffData, fc2.output_d);
		fullyConnectedBackward(fc2, n, fc2act.del_d);

		activationBackward(fc1act, 	n, fc2.del_d, fc1.output_d);
		fullyConnectedBackward(fc1, n, fc1act.del_d);		


		poolBackward(pool2,			n, fc1.del_d, conv2.output_d);
		convoluteBackward(conv2,	n, pool2.del_d);

		poolBackward(pool1,			n, conv2.del_d, conv1.output_d);


		// Update Weights
		fullyConnectedUpdateWeights(fc2, fc2act.del_d, fc1act.output_d);
		fullyConnectedUpdateWeights(fc1, fc1act.del_d,  pool1.output_d);

		convolutionalUpdateWeights(conv2, pool2.del_d, pool1.output_d);
		convolutionalUpdateWeights(conv1, pool1.del_d, image_data_d);

		checkCudaErrors( cudaFree(diffData) );
		return id;
	}
	/*
	int learn_example(value_type* image_data_d, 
						const Layer_t<value_type>& fc1,
						const Layer_t<value_type>& fc1act,
						const Layer_t<value_type>& fc2,
						const Layer_t<value_type>& fc2act,
						int target)
	{
		int n,c,h,w;
		
		int id = predict_example(image_data_d, fc1, fc1act, fc2, fc2act);

		//if (DEBUG) println("Performing backward propagation ...");
		n = h = w = 1; c = fc2act.outputs;

		value_type *diffData = NULL;
		resize(c, &diffData);
		checkCudaErrors( cudaMemcpy(diffData, fc2act.output_d, MSIZE(c), cudaMemcpyDeviceToDevice) );
		// getDiffData(fc2act, target, &diffData);
		getDiffDataD<<<1, c>>>(target, diffData);
		cudaDeviceSynchronize();

		// activationBackward(fc2act,	n, c, h, w, diffData, fc2.output_d);
		softmaxBackward(fc2act,		n, c, h, w, diffData, fc2.output_d);
		fullyConnectedBackward(fc2, n, c, h, w, fc2act.del_d);

		activationBackward(fc1act, 	n, c, h, w, fc2.del_d, fc1.output_d);
		// fullyConnectedBackward(fc1, n, c, h, w, fc1act.del_d);		


		// Update Weights
		fullyConnectedUpdateWeights(fc2, fc2act.del_d, fc1act.output_d);
		fullyConnectedUpdateWeights(fc1, fc1act.del_d,  image_data_d);

		checkCudaErrors( cudaFree(diffData) );
		return id;
	}
	*/

	static void load_mnist_data(value_type **training_data, value_type **testing_data,
		 value_type **training_target, value_type **testing_target,
		 int &total_train_size, int &total_test_size)
	{
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
				// println("Calculating file "<<fname<<"\t"<<file_size);
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
				// println("Reading file "<<fname<<" "<<file_size<<" "<<m);
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

/******************************************************************************
 * HELPER FUNCTIONS for main()
 *****************************************************************************/

void displayUsage()
{
	printf( "mnistCUDNN {<options>}\n");
	printf( "help                   : display this help\n");
	printf( "device=<int>           : set the device to run the sample\n");
	// printf( "image=<name>           : classify specific image\n");
}

template <typename value_type> 
void readImageToDevice(const char* fname, value_type **image_data_d){
	value_type imgData_h[N];
	readImage(fname, imgData_h);
	checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(N)) );
	checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, MSIZE(N), cudaMemcpyHostToDevice) );
}

/*
void run_alexnet()
{
	typedef MATRIX_DATA_TYPE value_type;
	// Define and initialize network
	const double base_learning_rate = 0.01;
	const double base_gamma = 0.001;
	const double base_power = 0.75;
	network_t<value_type> alexnet;
	Layer_t<value_type> conv1; 	conv1.initConvLayer("conv1", 1, 20, 5, IMAGE_H, IMAGE_W);

	Layer_t<value_type> pool1; 	pool1.initPoolLayer("pool1", 2, 2, conv1);

	Layer_t<value_type> conv2; 	conv2.initConvLayer("conv2", conv1.outputs, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride, conv1.outputs * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
	Layer_t<value_type> pool2; 	pool2.initPoolLayer("pool2", 2, 2, conv2);

	Layer_t<value_type> fc1;	fc1.initFCLayer(	"fc1", (conv2.outputs*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 500);
	Layer_t<value_type> fc1act; fc1act.initActLayer("fc1act", fc1.outputs);

	Layer_t<value_type> fc2; 	fc2.initFCLayer(	"fc2", fc1act.outputs, 10);

	Layer_t<value_type> fc2act; fc2act.initActLayer("fc2act", fc2.outputs);

	// Contains Training and Testing Examples
	value_type *train_data, *testing_data;
	value_type *train_target, *testing_target;

	// Read training data in tempraroy variables
	value_type *temp_training_data;
	value_type *temp_training_target;

	int total_train_data, total_test_data;
	alexnet.load_mnist_data(&temp_training_data, &testing_data, &temp_training_target, &testing_target, total_train_data, total_test_data);
	println("\n\nData Loaded. Training examples:"<<total_train_data/N<<" Testing examples:"<<total_test_data/N<<" Data dimension:"<<N);

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
			train_data[i*N+j] = temp_training_data[perm[i]*N+j];
		}
		train_target[i] = temp_training_target[perm[i]];
	}
	println("Training Examples shuffled.");

	// Free some variables
	delete [] temp_training_data;
	delete [] temp_training_target;
	delete [] perm;

	// Normalizing input data by dividing by 255
	for (int i=0; i<total_train_data; i++)
		train_data[i] /= 255;
	for (int i=0; i<total_test_data; i++)
		testing_data[i] /= 255;

	value_type* image_data_d = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(total_train_data)) );
	checkCudaErrors( cudaMemcpy(image_data_d, train_data, MSIZE(total_train_data), cudaMemcpyHostToDevice) );

	value_type* image_data_d2 = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d2, MSIZE(total_test_data)) );	
	checkCudaErrors( cudaMemcpy(image_data_d2, testing_data, MSIZE(total_test_data), cudaMemcpyHostToDevice) );

	// Try to load learned weights from file other wise start learning phase
	if (conv1.load() && conv2.load() && fc1.load() && fc2.load())
	{
		conv1.copyDataToDevice();
		conv2.copyDataToDevice();
		fc1.copyDataToDevice();
		fc2.copyDataToDevice();
		println("Weights from file loaded");
		// Testing Phase
		{
			print("\nTesting : ");
			std::clock_t    start;
			start = std::clock(); 
			int correct = 0;
			int n = total_test_data/N;
			
			for (int i=0; i<n; i++){
				value_type target = testing_target[i];
				value_type predicted = alexnet.predict_example(image_data_d2 + i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);
				
				if (target == predicted){
					correct++;
				}
				if (!DEBUG && i%1000==0) print("."<<std::flush);
				// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
			}
			println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
		}
	}
	else{
		println("\n **** Learning started ****");
		std::clock_t    start;
		start = std::clock(); 

		// Learn all examples till convergence
		int max_iterations = 50, iterations = 0, best_correct = 0;
		while(iterations++ < max_iterations){ // TODO: Use a better convergence criteria
			// Training Iteration
			{
				learning_rate = base_learning_rate*pow((1.0+base_gamma*(iterations-1)), -base_power);
				print("learning rate: "<<learning_rate<<" ");
				std::clock_t    start;
				start = std::clock();
				for (int i=0; i<m; i++){
					if (DEBUG) print("\n\n\n\n\n");
					value_type target = train_target[i];
					value_type predicted = alexnet.learn_example(image_data_d +i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act, target);
					if (DEBUG) getchar();
					else if (i%1000==0) print("."<<std::flush);
					//println("Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			}

			conv1.copyDataToHost();
			conv2.copyDataToHost();
			fc1.copyDataToHost();
			fc2.copyDataToHost();
			// Save the weights in a binary file
			if (conv1.save() && conv2.save() && fc1.save() && fc2.save())
				println("Weights Saved after "<<iterations<<" iterations.");

			// Testing Phase
			{
				print("\nTesting ("<<iterations<<") : ");
				std::clock_t    start;
				start = std::clock(); 
				int correct = 0;
				int n = total_test_data/N;
				
				for (int i=0; i<n; i++){
					value_type target = testing_target[i];
					value_type predicted = alexnet.predict_example(image_data_d2 + i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);
					
					if (target == predicted){
						correct++;
					}
					if (!DEBUG && i%1000==0) print("."<<std::flush);
					// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
				println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
				if (correct<best_correct){
					println("Accuracy started to decrease. Stopping Learning!! "<<correct-best_correct<<" misclassified.");
					break;
				}
				print("Correctly classified "<<(correct-best_correct)<<" new examples. ");
				best_correct = correct;
				conv1.copyDataToHost();
				conv2.copyDataToHost();
				fc1.copyDataToHost();
				fc2.copyDataToHost();
				// Save the weights in a binary file
				if (conv1.save() && conv2.save() && fc1.save() && fc2.save())
					println("Weights Saved.");
			}
		}
		
		println("\n **** Learning completed ****");
		println("Learning Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
	}
	checkCudaErrors( cudaFree(image_data_d2) );
	checkCudaErrors( cudaFree(image_data_d) );
}
*/

void run_lenet()
{
	typedef MATRIX_DATA_TYPE value_type;
	
	// Define and initialize network
	const double base_learning_rate = 0.01;
	const double base_gamma = 0.001;
	const double base_power = 0.75;
	
	network_t<value_type> lenet;

	Layer_t<value_type> conv1; 	conv1.initConvLayer("conv1", 1, 20, 5, 1, IMAGE_H, IMAGE_W);

	Layer_t<value_type> pool1; 	pool1.initPoolLayer("pool1", 2, 2, conv1);

	Layer_t<value_type> conv2; 	conv2.initConvLayer("conv2", pool1.kernel_dim, 50, 5, 1, pool1.out_width, pool1.out_height, pool1.outputs);

	Layer_t<value_type> pool2; 	pool2.initPoolLayer("pool2", 2, 2, conv2);

	Layer_t<value_type> fc1;	fc1.initFCLayer(	"fc1", pool2.outputs, 500);

	Layer_t<value_type> fc1act; fc1act.initActLayer("fc1act", fc1.outputs);

	Layer_t<value_type> fc2; 	fc2.initFCLayer(	"fc2", fc1act.outputs, 10);

	Layer_t<value_type> fc2act; fc2act.initActLayer("fc2act", fc2.outputs);

	// Contains Training and Testing Examples
	value_type *train_data, *testing_data;
	value_type *train_target, *testing_target;

	// Read training data in tempraroy variables
	value_type *temp_training_data;
	value_type *temp_training_target;

	int total_train_data, total_test_data;
	lenet.load_mnist_data(&temp_training_data, &testing_data, &temp_training_target, &testing_target, total_train_data, total_test_data);
	println("\n\nData Loaded. Training examples:"<<total_train_data/N<<" Testing examples:"<<total_test_data/N<<" Data dimension:"<<N);

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
			train_data[i*N+j] = temp_training_data[perm[i]*N+j];
		}
		train_target[i] = temp_training_target[perm[i]];
	}
	println("Training Examples shuffled.");

	// Free some variables
	delete [] temp_training_data;
	delete [] temp_training_target;
	delete [] perm;

	// Normalizing input data by dividing by 255
	for (int i=0; i<total_train_data; i++)
		train_data[i] /= 255;
	for (int i=0; i<total_test_data; i++)
		testing_data[i] /= 255;

	value_type* image_data_d = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(total_train_data)) );
	checkCudaErrors( cudaMemcpy(image_data_d, train_data, MSIZE(total_train_data), cudaMemcpyHostToDevice) );

	value_type* image_data_d2 = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d2, MSIZE(total_test_data)) );	
	checkCudaErrors( cudaMemcpy(image_data_d2, testing_data, MSIZE(total_test_data), cudaMemcpyHostToDevice) );

	// Try to load learned weights from file other wise start learning phase
	if (conv1.load() && conv2.load() && fc1.load() && fc2.load())
	{
		conv1.copyDataToDevice();
		conv2.copyDataToDevice();
		fc1.copyDataToDevice();
		fc2.copyDataToDevice();
		println("Weights from file loaded");
		// Testing Phase
		{
			print("\nTesting : ");
			std::clock_t    start;
			start = std::clock(); 
			int correct = 0;
			int n = total_test_data/N;
			
			for (int i=0; i<n; i++){
				value_type target = testing_target[i];
				value_type predicted = lenet.predict_example(image_data_d2 + i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);
				
				if (target == predicted){
					correct++;
				}
				if (!DEBUG && i%1000==0) print("."<<std::flush);
				// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
			}
			println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
		}
	}
	else{
		println("\n **** Learning started ****");
		std::clock_t    start;
		start = std::clock(); 

		// Learn all examples till convergence
		int max_iterations = 50, iterations = 0, best_correct = 0;
		while(iterations++ < max_iterations){ // TODO: Use a better convergence criteria
			// Training Iteration
			{
				learning_rate = base_learning_rate*pow((1.0+base_gamma*(iterations-1)), -base_power);
				print("\n\nLearning ("<<iterations<<") rate: "<<learning_rate<<" ");
				std::clock_t    start;
				start = std::clock();
				for (int i=0; i<m; i++){
					if (DEBUG) print("\n\n\n\n\n");
					value_type target = train_target[i];
					value_type predicted = lenet.learn_example(image_data_d +i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act, target);
					if (DEBUG) getchar();
					else if (i%1000==0) print("."<<std::flush);
					//println("Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			}

			conv1.copyDataToHost();
			conv2.copyDataToHost();
			fc1.copyDataToHost();
			fc2.copyDataToHost();
			// Save the weights in a binary file
			if (conv1.save() && conv2.save() && fc1.save() && fc2.save())
				println("Weights Saved after "<<iterations<<" iterations.");

			// Testing Phase
			{
				print("\nTesting ("<<iterations<<") : ");
				std::clock_t    start;
				start = std::clock(); 
				int correct = 0;
				int n = total_test_data/N;
				
				for (int i=0; i<n; i++){
					value_type target = testing_target[i];
					value_type predicted = lenet.predict_example(image_data_d2 + i*N, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);
					
					if (target == predicted){
						correct++;
					}
					if (!DEBUG && i%1000==0) print("."<<std::flush);
					// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
				println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
				if (correct<best_correct){
					println("Accuracy started to decrease. Stopping Learning!! "<<correct-best_correct<<" misclassified.");
					break;
				}
				print("Correctly classified "<<(correct-best_correct)<<" new examples. ");
				best_correct = correct;
				conv1.copyDataToHost();
				conv2.copyDataToHost();
				fc1.copyDataToHost();
				fc2.copyDataToHost();
				// Save the weights in a binary file
				if (conv1.save() && conv2.save() && fc1.save() && fc2.save())
					println("Weights Saved.");
			}
		}
		
		println("\n **** Learning completed ****");
		println("Learning Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
	}
	checkCudaErrors( cudaFree(image_data_d2) );
	checkCudaErrors( cudaFree(image_data_d) );
}

/*
void run_mnist()
{
	typedef MATRIX_DATA_TYPE value_type;
	// Define and initialize network
	const double base_learning_rate = 0.01;
	const double base_gamma = 0.0001;
	const double base_power = 0.75;
	network_t<value_type> mnist;

	Layer_t<value_type> fc1;	fc1.initFCLayer(	"fc1", N, 100);
	Layer_t<value_type> fc1act; fc1act.initActLayer("fc1act", fc1.outputs);
	Layer_t<value_type> fc2; 	fc2.initFCLayer(	"fc2", fc1act.outputs, 10);
	Layer_t<value_type> fc2act; fc2act.initActLayer("fc2act", fc2.outputs);

	// Contains Training and Testing Examples
	value_type *train_data, *testing_data;
	value_type *train_target, *testing_target;

	// Read training data in tempraroy variables
	value_type *temp_training_data;
	value_type *temp_training_target;

	int total_train_data, total_test_data;
	mnist.load_mnist_data(&temp_training_data, &testing_data, &temp_training_target, &testing_target, total_train_data, total_test_data);
	println("\n\nData Loaded. Training examples:"<<total_train_data/N<<" Testing examples:"<<total_test_data/N<<" Data dimension:"<<N);

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
			train_data[i*N+j] = temp_training_data[perm[i]*N+j];
		}
		train_target[i] = temp_training_target[perm[i]];
	}
	println("Training Examples shuffled.");

	// Free some variables
	delete [] temp_training_data;
	delete [] temp_training_target;
	delete [] perm;

	// Normalizing input data by dividing by 255
	for (int i=0; i<total_train_data; i++)
		train_data[i] /= 255;
	for (int i=0; i<total_test_data; i++)
		testing_data[i] /= 255;

	value_type* image_data_d = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(total_train_data)) );
	checkCudaErrors( cudaMemcpy(image_data_d, train_data, MSIZE(total_train_data), cudaMemcpyHostToDevice) );

	value_type* image_data_d2 = NULL;
	checkCudaErrors( cudaMalloc(&image_data_d2, MSIZE(total_test_data)) );	
	checkCudaErrors( cudaMemcpy(image_data_d2, testing_data, MSIZE(total_test_data), cudaMemcpyHostToDevice) );

	// Try to load learned weights from file other wise start learning phase
	if (fc1.load() && fc2.load())
	{
		fc1.copyDataToDevice();
		fc2.copyDataToDevice();
		println("Weights from file loaded");
		// Testing Phase
		{
			print("\nTesting : ");
			std::clock_t    start;
			start = std::clock(); 
			int correct = 0;
			int n = total_test_data/N;
			
			for (int i=0; i<n; i++){
				value_type target = testing_target[i];
				value_type predicted = mnist.predict_example(image_data_d2 + i*N, fc1, fc1act, fc2, fc2act);
				
				if (target == predicted){
					correct++;
				}
				if (!DEBUG && i%1000==0) print("."<<std::flush);
				// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
			}
			println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
		}
	}
	else{
		println("\n **** Learning started ****");
		std::clock_t    start;
		start = std::clock(); 

		// Learn all examples till convergence
		int max_iterations = 50, iterations = 0, best_correct = 0;
		while(iterations++ < max_iterations ){ // TODO: Use a better convergence criteria
			// Training Iteration
			{
				learning_rate = base_learning_rate*pow((1.0+base_gamma*(iterations-1)), -base_power);
				print("learning rate: "<<learning_rate<<" ");
				std::clock_t    start;
				start = std::clock();
				for (int i=0; i<m; i++){
					if (DEBUG) print("\n\n\n\n\n");
					value_type target = train_target[i];
					value_type predicted = mnist.learn_example(image_data_d +i*N, fc1, fc1act, fc2, fc2act, target);
					if (DEBUG) getchar();
					else if (i%1000==0) print("."<<std::flush);
					//println("Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			}

			// Testing Phase
			{
				print("\nTesting ("<<iterations<<") : ");
				std::clock_t    start;
				start = std::clock(); 
				int correct = 0;
				int n = total_test_data/N;
				
				for (int i=0; i<n; i++){
					value_type target = testing_target[i];
					value_type predicted = mnist.predict_example(image_data_d2 + i*N, fc1, fc1act, fc2, fc2act);
					
					if (target == predicted){
						correct++;
					}
					if (!DEBUG && i%1000==0) print("."<<std::flush);
					// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
				println("\tTime: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
				println("Accuracy: "<<((100.0 * correct)/n)<<" %\t\tCorrectly predicted "<<correct<<" examples out of "<<n);
				if (correct<best_correct){
					println("Accuracy started to decrease. Stopping Learning!! "<<correct-best_correct<<" misclassified.");
					break;
				}
				print("Correctly classified "<<(correct-best_correct)<<" new examples. ");
				best_correct = correct;
				fc1.copyDataToHost();
				fc2.copyDataToHost();
				// Save the weights in a binary file
				if (fc1.save() && fc2.save())
					println("Weights Saved.");
			}
		}
		
		println("\n **** Learning completed ****");
		println("Total Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
	}
	checkCudaErrors( cudaFree(image_data_d2) );
	checkCudaErrors( cudaFree(image_data_d) );
}
*/


/******************************************************************************
 * MAIN() function
 *****************************************************************************/

int main(int argc, char *argv[])
{   
	// Print Usage if help is in the arguments
	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		displayUsage();
		exit(EXIT_WAIVED); 
	}

	// Print Library and Device stats
	int version = (int)cudnnGetVersion();
	printf("\n\nCuDNN Version : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
	printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
	showDevices();

	// If device argument is provided then set device (device=1)
	int device = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		checkCudaErrors( cudaSetDevice(device) );
	}
	println("Using device " << device);


	srand(time(NULL));

	bool alexnet = true;
	if (alexnet)
	{
		run_lenet();
	} else 
	{
		// run_mnist();
	}

	// Reset device and exit gracefully
	cudaDeviceReset();
	exit(EXIT_SUCCESS);        
}
