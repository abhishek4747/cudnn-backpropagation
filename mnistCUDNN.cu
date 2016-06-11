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

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
	template <class T>
	value_type operator()(T x) {return value_type(x);}
};

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
	// conversion
	Convert<value_type> fromReal;
	for (int i = 0; i < size; i++)
	{
		data_h[i] = fromReal(data_tmp[i]);
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
	// Take care of half precision
	Convert<value_type> fromReal;
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
			imgData_h[idx] = fromReal(*(oHostSrc.data() + idx) / double(255));
		}
	} 
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
	typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
	value_type *vec;
	vec = new value_type[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, MSIZE(size), cudaMemcpyDeviceToHost);
	Convert<real_type> toReal;
	std::cout.precision(5);
	std::cout.setf( std::ios::fixed, std::ios::floatfield );
	for (int i = 0; i < size; i++)
	{
		print(toReal(vec[i]) << " ");
	}
	println(" ");
	delete [] vec;
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

	int inputs, outputs, kernel_dim; // linear dimension (i.e. size is kernel_dim * kernel_dim)
	int  w_size, b_size;

	int in_height, in_width;
	int out_height, out_width;

	value_type *data_h, 	*data_d;
	value_type *bias_h, 	*bias_d;

	value_type *output_d,	*del_d;

	// Convolutional Layer
	cudnnConvolutionDescriptor_t convDesc;
	cudnnTensorDescriptor_t convTensor, convBiasTensor;
	cudnnFilterDescriptor_t convFilterDesc;
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
	cudnnConvolutionFwdAlgo_t convAlgo;

	// Pooling Layer
	cudnnPoolingDescriptor_t poolDesc;
	cudnnTensorDescriptor_t poolSrcTensor, poolDstTensor, poolBiasTensor;
	cudnnFilterDescriptor_t poolFilterDesc;
	int size, stride;
	
	// Fully Connected Layer


	// Activation Layer


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
	};

	~Layer_t()
	{
		if (data_h != NULL) 	delete [] data_h;
		if (bias_h != NULL) 	delete [] bias_h;

		if (data_d != NULL) 	checkCudaErrors( cudaFree(data_d) );
		if (bias_d != NULL) 	checkCudaErrors( cudaFree(bias_d) );
		if (output_d != NULL) 	checkCudaErrors( cudaFree(output_d) );
		if (del_d != NULL) 		checkCudaErrors( cudaFree(del_d) );
	}

	size_t initConvLayer(std::string _layername, int _inputs, int _outputs, int _kernel_dim, int _in_height, int _in_width, int _d_size=0)
	{
		layerType 	= CONV_LAYER;
		layername 	= _layername;
		inputs 		= _inputs;
		outputs 	= _outputs;
		kernel_dim 	= _kernel_dim;
		in_width 	= _in_width;
		in_height 	= _in_height;
		out_width 	= in_width - kernel_dim + 1;			// Stride is 1	// TODO: Make Generic
		out_height 	= in_height - kernel_dim + 1;			// Stride is 1	// TODO: Make Generic
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
		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(outputs*out_height*out_width)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(_d_size)) );

		copyDataToDevice();

		checkCUDNN(cudnnCreateTensorDescriptor(&convTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&convBiasTensor));

		size_t sizeInBytes = 0;

		int n = 1;
		int c = inputs;
		int h = in_height;
		int w = in_width;

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              tensorFormat,
                                              dataType,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetTensor4dDescriptor(convBiasTensor,
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
                                                   1, 1,	//	stride
                                                   1, 1,	// 	upscaling
                                                   CUDNN_CROSS_CORRELATION));
        // Find dimension of convolution output
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                         srcTensorDesc,
                                                         convFilterDesc,
                                                         &n, &c, &h, &w));
		
		checkCUDNN(cudnnSetTensor4dDescriptor(convTensor,
                                              tensorFormat,
                                              dataType,
                                              n, c,
                                              h, w));


        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              tensorFormat,
                                              dataType,
                                              n, c,
                                              h, w));
        cudnnHandle_t cudnnHandle;
		checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                       srcTensorDesc,
                                                       convFilterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &convAlgo));
        
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                           srcTensorDesc,
                                                           convFilterDesc,
                                                           convDesc,
                                                           dstTensorDesc,
                                                           convAlgo,
                                                           &sizeInBytes));
        // println("Best Conv Algo "<<convAlgo);
        convAlgo = (cudnnConvolutionFwdAlgo_t)convAlgo;
        checkCUDNN( cudnnDestroy(cudnnHandle) );
		return sizeInBytes;
	}

	void initPoolLayer(std::string _layername, int _size, int _stride, const Layer_t<value_type>& conv)
	{
		layerType 	= POOL_LAYER;
		layername 	= _layername;
		size 		= _size;
		stride 		= _stride;
		w_size		= 0;
		b_size		= conv.outputs*(conv.out_width / stride) * (conv.out_height / stride);
		outputs 	= b_size;
		inputs  	= conv.outputs*conv.out_width*conv.out_height; 
		
		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(inputs)) );

		checkCUDNN(cudnnCreateTensorDescriptor(&poolSrcTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&poolDstTensor));
		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

		// checkCUDNN(cudnnSetTensor4dDescriptor(poolSrcTensor,
  //                                             tensorFormat,
  //                                             dataType,
  //                                             1, conv.outputs,
  //                                             conv.out_height / stride,
		// 									  conv.out_width / stride));


		// checkCUDNN(cudnnSetTensor4dDescriptor(poolDstTensor,
  //                                             tensorFormat,
  //                                             dataType,
  //                                             1, conv.outputs,
  //                                             conv.out_height / stride,
		// 									  conv.out_width / stride));

		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
											   CUDNN_POOLING_MAX,
											   CUDNN_PROPAGATE_NAN,
											   size, size,
											   0, 0,
											   stride, stride));
	}

	void initFCLayer(std::string _layername, int _inputs, int _outputs){
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
		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(inputs)) );

		copyDataToDevice();
	}

	void initLayer(std::string _layername, LayerType _layerType, int _outputs){
		layerType 	= _layerType;
		layername 	= _layername;
		inputs 		= _outputs;
		outputs 	= _outputs;
		kernel_dim 	= 1;
		w_size 		= 0;
		b_size 		= 0;
		
		checkCudaErrors( cudaMalloc(&output_d, 	MSIZE(outputs)) );
		checkCudaErrors( cudaMalloc(&del_d, 	MSIZE(inputs)) );
	}

	void destroyConvLayer(){
		checkCUDNN(cudnnDestroyTensorDescriptor(convTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(convFilterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(convBiasTensor));
	}

	void destoryPoolLayer(){
		checkCUDNN(cudnnDestroyTensorDescriptor(poolSrcTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(poolDstTensor));
		checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
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
	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
	}
};


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
 * network_t class : contains all learning functions
 *****************************************************************************/

template <class value_type>
class network_t
{
	typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
	int convAlgorithm;
	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc, srcDiffTensorDesc, dstDiffTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnPoolingDescriptor_t     poolingDesc;
	cudnnActivationDescriptor_t  activDesc;
	cudnnLRNDescriptor_t   normDesc;
	cublasHandle_t cublasHandle;
	cudnnConvolutionFwdAlgo_t convAlgo;

	void createHandles()
	{
		checkCUDNN( cudnnCreate(&cudnnHandle) );
		checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
		checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
		checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
		checkCUDNN( cudnnCreateTensorDescriptor(&srcDiffTensorDesc) );
		checkCUDNN( cudnnCreateTensorDescriptor(&dstDiffTensorDesc) );
		checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
		checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
		checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
		checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
		checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );
	
		checkCublasErrors( cublasCreate(&cublasHandle) );
		convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	}

	void destroyHandles()
	{
		checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
		checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
		checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
		checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
		checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(srcDiffTensorDesc) );
		checkCUDNN( cudnnDestroyTensorDescriptor(dstDiffTensorDesc) );
		checkCUDNN( cudnnDestroy(cudnnHandle) );

		checkCublasErrors( cublasDestroy(cublasHandle) );
	}
  public:
	network_t()
	{
		convAlgorithm = -1;
		switch (sizeof(value_type))
		{
			case 4 : dataType = CUDNN_DATA_FLOAT; break;
			case 8 : dataType = CUDNN_DATA_DOUBLE; break;
			default : FatalError("Unsupported data type");
		}
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
		checkCudaErrors( cudaMalloc(data, MSIZE(size)) );
	}
	
	void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
	{
		convAlgorithm = (int) algo;
	}
	
	void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
	{
		setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(1);
		checkCUDNN( cudnnAddTensor( cudnnHandle, 
									&alpha, biasTensorDesc,
									layer.bias_d,
									&beta,
									dstTensorDesc,
									data) );
	}

	void fullyConnectedForward(const Layer_t<value_type>& layer,
						  int& n, int& c, int& h, int& w,
						  value_type* srcData)
	{
		if (n != 1)
		{
			FatalError("Not Implemented"); 
		}
		// println("fullyConnectedforward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		int dim_x = c*h*w;
		int dim_y = layer.outputs;
		// resize(dim_y, &layer.output_d);

		scaling_type alpha = scaling_type(1), beta = scaling_type(1);
		// place bias into dstData
		checkCudaErrors( cudaMemcpy(layer.output_d, layer.bias_d, MSIZE(dim_y), cudaMemcpyDeviceToDevice) );
		
		// gemv(cublasHandle, dim_x, dim_y, alpha,
		// 		layer.data_d, srcData, beta, layer.output_d);
		checkCublasErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_T,
                                  dim_x, dim_y,
                                  &alpha,
                                  layer.data_d, dim_x,
                                  srcData, 1,
                                  &beta,
                                  layer.output_d, 1) );    

		h = 1; w = 1; c = dim_y;
	}

	void convoluteForward(const Layer_t<value_type>& conv,
						  int& n, int& c, int& h, int& w,
						  value_type* srcData)
	{

		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims] = {n,c,h,w};
		const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
										conv.kernel_dim, conv.kernel_dim};
									   
		checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
											  dataType,
											  tensorFormat,
											  tensorDims,
											  filterDimA) );
 
		const int convDims = 2;
		int padA[convDims] = {0,0};
		int filterStrideA[convDims] = {1,1};
		int upscaleA[convDims] = {1,1};
		cudnnDataType_t  convDataType = dataType;
		checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
													convDims,
													padA,
													filterStrideA,
													upscaleA,
													CUDNN_CROSS_CORRELATION,
													convDataType) );
		// find dimension of convolution output
		checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
												srcTensorDesc,
												filterDesc,
												tensorDims,
												tensorOuputDimA) );
		n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

		// println("convoluteForward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

		// resize(n*c*h*w, &(conv.output_d));
		if (DEBUG) printDeviceVector("Conv Weights:\n", conv.w_size, conv.data_d);
		if (DEBUG) printDeviceVector("Conv Bias:\n", conv.b_size, conv.bias_d);
		size_t sizeInBytes=0;
		void* workSpace=NULL;
		checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
												srcTensorDesc,
												filterDesc,
												convDesc,
												dstTensorDesc,
												convAlgo,
												&sizeInBytes) );
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
		}
		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
		checkCUDNN( cudnnConvolutionForward(cudnnHandle,
											  &alpha,
											  srcTensorDesc,
											  srcData,
											  filterDesc,
											  conv.data_d,
											  convDesc,
											  convAlgo,
											  workSpace,
											  sizeInBytes,
											  &beta,
											  dstTensorDesc,
											  conv.output_d) );
		addBias(dstTensorDesc, conv, c, conv.output_d);
		if (DEBUG) printDeviceVector("Conv Output:\n", conv.outputs*conv.out_height*conv.out_width, conv.output_d);
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}
	}

	void convoluteBackward(const Layer_t<value_type>& layer,
							int& n, int& c, int& h, int& w,
							value_type* diffData)
	{
		// println("convoluteBackward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);

		size_t sizeInBytes = 0;
		void* workSpace=NULL;
		cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

		checkCUDNN( cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
														layer.convFilterDesc,
														layer.convTensor,
														layer.convDesc,
														layer.srcTensorDesc,
														algo,
														&sizeInBytes
														));
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
		}
		value_type alpha = value_type(1);
		value_type beta  = value_type(0);
		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, 
												&alpha, 
												layer.convFilterDesc, layer.data_d, 
												layer.convTensor, diffData, 
												layer.convDesc, algo,
												workSpace, sizeInBytes,
												&beta, 
												layer.srcTensorDesc, layer.del_d));
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}
	}

	void poolForward(const Layer_t<value_type>& layer,
					  int& n, int& c, int& h, int& w,
					  value_type* srcData)
	{
		// const int poolDims = 2;
		// int windowDimA[poolDims] = {2,2};
		// int paddingA[poolDims] = {0,0};
		// int strideA[poolDims] = {2,2};
		// checkCUDNN( cudnnSetPoolingNdDescriptor(layer.poolDesc,
		// 										CUDNN_POOLING_MAX,
		// 										CUDNN_PROPAGATE_NAN,
		// 										poolDims,
		// 										windowDimA,
		// 										paddingA,
		// 										strideA ) );

		setTensorDesc((cudnnTensorDescriptor_t&)layer.poolSrcTensor, tensorFormat, dataType, n, c, h, w);        

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims] = {n,c,h,w};
		checkCUDNN( cudnnGetPoolingNdForwardOutputDim(layer.poolDesc,
													layer.poolSrcTensor,
													tensorDims,
													tensorOuputDimA) );
		n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
		// println("poolingForward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		
		setTensorDesc((cudnnTensorDescriptor_t&)layer.poolDstTensor, tensorFormat, dataType, n, c, h, w);  
	 
		// resize(n*c*h*w, &(layer.output_d));
		if (DEBUG) printDeviceVector("Pooling Input:\n", layer.inputs, layer.output_d);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		checkCUDNN( cudnnPoolingForward(cudnnHandle,
										  layer.poolDesc,
										  &alpha,
										  layer.poolSrcTensor,
										  srcData,
										  &beta,
										  layer.poolDstTensor,
										  layer.output_d) );
		if (DEBUG) printDeviceVector("Pooling Output:\n", layer.outputs, layer.output_d);
	}

	void poolBackward(const Layer_t<value_type>& layer,
						int& n, int& c, int& h, int& w,
						value_type* diffData, value_type* srcData)
	{
		if (DEBUG) println("poolingback::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);

		value_type alpha = value_type(1.0);
		value_type beta  = value_type(0.0);
		if (DEBUG) printDeviceVector("Pooling back Input: ", layer.outputs, srcData);
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, 
											layer.poolDesc, 
											&alpha, 
											layer.poolDstTensor, layer.output_d, 
											layer.poolDstTensor, diffData,
											layer.poolSrcTensor, srcData, 
											&beta, 
											layer.poolSrcTensor, layer.del_d));
		if (DEBUG) printDeviceVector("Pooling back Output: ", layer.inputs, layer.del_d);

	}

	void softmaxForward(const Layer_t<value_type>& layer, 
						int &n, int &c, int &h, int &w, value_type* srcData)
	{
		// resize(n*c*h*w, &(layer.output_d));
		// println("softmaxForward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
		checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
										  CUDNN_SOFTMAX_ACCURATE ,
										  CUDNN_SOFTMAX_MODE_CHANNEL,
										  &alpha,
										  srcTensorDesc,
										  srcData,
										  &beta,
										  dstTensorDesc,
										  layer.output_d) );
	}

	void getDiffData(const Layer_t<value_type>& layer, int target, value_type** diffData){
		resize(layer.outputs, diffData);
		value_type outputh[layer.outputs];
		checkCudaErrors( cudaMemcpy(outputh, layer.output_d, MSIZE(layer.outputs), cudaMemcpyDeviceToHost) );
		for (int i=0; i<layer.outputs; i++){
			if (i==target)
				outputh[i] = 1 - outputh[i];
			else
				outputh[i] = 0 - outputh[i];
		}
		checkCudaErrors( cudaMemcpy(*diffData, outputh, MSIZE(layer.outputs), cudaMemcpyHostToDevice) );
	}

	void softmaxBackward(const Layer_t<value_type>& layer, 
						int &n, int &c, int &h, int &w, 						
						value_type* diffData, value_type* srcData)
	{
		// println("softmaxBackward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
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
		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
		checkCUDNN( cudnnSoftmaxBackward(cudnnHandle,
										  CUDNN_SOFTMAX_ACCURATE ,
										  CUDNN_SOFTMAX_MODE_CHANNEL,
										  &alpha,
										  srcTensorDesc,
										  layer.output_d,
										  srcDiffTensorDesc,
										  diffData,
										  &beta,
										  dstTensorDesc,
										  layer.del_d) );
	}
	void lrnForward(int &n, int &c, int &h, int &w, value_type* srcData, value_type** dstData)
	{
		unsigned lrnN = 5;
		double lrnAlpha, lrnBeta, lrnK;
		lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
		checkCUDNN( cudnnSetLRNDescriptor(normDesc,
											lrnN,
											lrnAlpha,
											lrnBeta,
											lrnK) );

		resize(n*c*h*w, dstData);

		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
		checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
											normDesc,
											CUDNN_LRN_CROSS_CHANNEL_DIM1,
											&alpha,
											srcTensorDesc,
											srcData,
											&beta,
											dstTensorDesc,
											*dstData) );
	}

	void activationForward(const Layer_t<value_type>& layer, 
							int &n, int &c, int &h, int &w, value_type* srcData)
	{
		// println("activationForward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		checkCUDNN( cudnnSetActivationDescriptor(activDesc,
												CUDNN_ACTIVATION_SIGMOID,
												CUDNN_PROPAGATE_NAN,
												0.0) );
	
		// resize(n*c*h*w, &(layer.output_d));

		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
		checkCUDNN( cudnnActivationForward(cudnnHandle,
											activDesc,
											&alpha,
											srcTensorDesc,
											srcData,
											&beta,
											dstTensorDesc,
											layer.output_d) );    
	}

	void fullyConnectedBackward(const Layer_t<value_type>& layer,
								int &n, int &c, int &h, int &w, value_type* srcData)
	{
		// println("fullyConnectedBack::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		value_type alpha = value_type(1), beta = value_type(0);
		checkCudaErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_N,
									  layer.inputs, layer.outputs,
									  &alpha,
									  layer.data_d, layer.inputs,
									  srcData, 1,
									  &beta,
									  layer.del_d, 1) );
		c = layer.inputs;
	}

	void activationBackward(const Layer_t<value_type>& layer,
							int &n, int &c, int &h, int &w, 
							value_type *srcDiffData, value_type* srcData)
	{
		// println("activationBackward::\tn:"<<n<<"\tc:"<<c<<"\th:"<<h<<"\tw:"<<w);
		checkCUDNN( cudnnSetActivationDescriptor(activDesc,
												CUDNN_ACTIVATION_SIGMOID,
												CUDNN_PROPAGATE_NAN,
												0.0) );
		// resize(n*c*h*w, dstDiffData);
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
											activDesc, // RELU
											&alpha,
											srcTensorDesc,
											layer.output_d,
											srcDiffTensorDesc,
											srcDiffData,
											dstTensorDesc,
											srcData,
											&beta,
											dstDiffTensorDesc,
											layer.del_d
											) );    
	}

	void fullyConnectedUpdateWeights(const Layer_t<value_type>& layer, value_type* diffData, value_type* srcData){
		int dim_x = layer.inputs;
		int dim_y = layer.outputs;
		int dim_z = 1;
		value_type* dstData = NULL;
		resize(dim_x*dim_y, &dstData);

		value_type alpha = value_type(1), beta = value_type(0);
		// checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, MSIZE(ip.outputs), cudaMemcpyDeviceToDevice) );
		//if (DEBUG) printDeviceVector("last_input: \n", layer.inputs, last_input);
		//if (DEBUG) printDeviceVector("del_W: \n", layer.outputs, layer.del_d);
		
		checkCudaErrors( CUBLAS_GEMM(cublasHandle, 
									  CUBLAS_OP_N, CUBLAS_OP_N,
									  dim_x, dim_y, dim_z,
									  &alpha,
									  srcData, dim_x,
									  diffData, dim_z,
									  &beta,
									  dstData, dim_x) );
		
		// if (DEBUG) printDeviceVector("\tdelta_W (del_W*hidden_input): \n", layer.inputs*layer.outputs, dstData);

		alpha = value_type(0.1); // learning rate
		beta = value_type(1); 
		//checkCudaErrors( cublasDscal(cublasHandle, ip.inputs*ip.outputs, &alpha, ip.data_d, 1); 
		const value_type* B = layer.data_d;
		// C = α op ( A ) + β * C
		// C = 0.1 * delta_W2 + C
		// if (DEBUG) printDeviceVector("\tW = W + 0.1*delta_W: old\n", dim_x*dim_y, layer.data_d);
		
		checkCudaErrors( CUBLAS_GEAM(cublasHandle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										dim_x, dim_y,
										&alpha,
										dstData, dim_x,
										&beta,
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
										&alpha,
										diffData, dim_x,
										&beta,
										B2, dim_x,
										layer.bias_d, dim_x) );
		// if (DEBUG) printDeviceVector("\tB:\n", layer.outputs, layer.bias_d);

		checkCudaErrors( cudaFree(dstData));
	}

	void convolutionalUpdateWeights(const Layer_t<value_type>& layer, value_type* diffData, value_type* srcData){
		value_type alpha = value_type(1);
		value_type beta  = value_type(0.0);
		cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

		if (DEBUG) println("Convolutional Update Weights:");

		value_type *gconvB = NULL, *gconvW = NULL;
		resize(layer.outputs, &gconvB);
		resize(layer.w_size, &gconvW);
		
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, 
												&alpha, 
												layer.convTensor, diffData, 
												&beta, 
												layer.convBiasTensor, gconvB));

		if (DEBUG) printDeviceVector(" gconvB: ", layer.outputs, gconvB);

		size_t sizeInBytes=0;
		void* workSpace=NULL;
		checkCUDNN( cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
												layer.srcTensorDesc,
												layer.convTensor,
												layer.convDesc,
												layer.convFilterDesc,
												algo,
												&sizeInBytes));
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
		}
		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, 
												&alpha, 
												layer.srcTensorDesc, srcData, 
												layer.convTensor, diffData, 
												layer.convDesc, algo,
												workSpace, sizeInBytes,
												&beta, 
												layer.convFilterDesc, gconvW));
		if (sizeInBytes!=0)
		{
		  checkCudaErrors( cudaFree(workSpace) );
		}

		if (DEBUG) printDeviceVector(" gconvW: ", layer.w_size, gconvW);

		alpha = value_type(0.1); // learning rate
		checkCudaErrors(cublasDaxpy(cublasHandle, 
									layer.outputs*layer.inputs*layer.kernel_dim*layer.kernel_dim,
									&alpha, 
									gconvW, 1, 
									layer.data_d, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, 
									layer.outputs,
									&alpha, 
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
		int n, c, h, w;
		// if (DEBUG) println("Performing forward propagation ...");

		n = c = 1; h = IMAGE_H; w = IMAGE_W;

		convoluteForward(conv1, n, c, h, w, image_data_d);
		poolForward(pool1, 		n, c, h, w, conv1.output_d);

		convoluteForward(conv2, n, c, h, w, pool1.output_d);
		poolForward(pool2, 		n, c, h, w, conv2.output_d);

		fullyConnectedForward(fc1, 	n, c, h, w, pool2.output_d);
		activationForward(fc1act, 	n, c, h, w, fc1.output_d);
		
		// lrnForward(n, c, h, w, srcData, &dstData);

		fullyConnectedForward(fc2, 	n, c, h, w, fc1act.output_d);
		// activationForward(fc2act, 	n, c, h, w, fc2.output_d);
		softmaxForward(fc2act, 	n, c, h, w, fc2.output_d);

		const int max_digits = fc2act.outputs;
		
		// Take care of half precision
		Convert<scaling_type> toReal;
		value_type result[max_digits];
		checkCudaErrors( cudaMemcpy(result, fc2act.output_d, MSIZE(max_digits), cudaMemcpyDeviceToHost) );
		int id = 0;
		for (int i = 1; i < max_digits; i++)
		{
			if (toReal(result[id]) < toReal(result[i])) id = i;
		}

		return id;
	}
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
		int n,c,h,w;
		
		int id = predict_example(image_data_d, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);

		//if (DEBUG) println("Performing backward propagation ...");
		n = h = w = 1; c = fc2act.outputs;

		value_type *diffData = NULL;
		
		getDiffData(fc2act, target, &diffData);

		// activationBackward(fc2act,	n, c, h, w, diffData, fc2.output_d);
		softmaxBackward(fc2act,		n, c, h, w, diffData, fc2.output_d);
		fullyConnectedBackward(fc2, n, c, h, w, fc2act.del_d);

		activationBackward(fc1act, 	n, c, h, w, fc2.del_d, fc1.output_d);
		fullyConnectedBackward(fc1, n, c, h, w, fc1act.del_d);		


		poolBackward(pool2,			n, c, h, w, fc1.del_d, conv2.output_d);
		convoluteBackward(conv2,	n, c, h, w, pool2.del_d);

		poolBackward(pool1,			n, c, h, w, conv2.del_d, conv1.output_d);


		// Update Weights
		fullyConnectedUpdateWeights(fc2, fc2act.del_d, fc1act.output_d);
		fullyConnectedUpdateWeights(fc1, fc1act.del_d,  pool1.output_d);

		convolutionalUpdateWeights(conv2, pool2.del_d, pool1.output_d);
		convolutionalUpdateWeights(conv1, pool1.del_d, image_data_d);

		checkCudaErrors( cudaFree(diffData) );
		return id;
	}

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

double *makeDiffData(int m, int c) {
  double *diff = (double *) calloc(m * c, sizeof(double));
  for (int j = 0; j < m; j++) {
    int cs = 4;//rand() % c;
    printf("%d cs: %d\n", j, cs);
    for (int i = 0; i < c; i++)
      diff[j * c + i] = cs == i ? -1 / (double) m : 0;
  }
  return diff;
}

template <typename value_type> 
void readImageToDevice(const char* fname, value_type **image_data_d){
	value_type imgData_h[N];
	readImage(fname, imgData_h);
	checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(N)) );
	checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, MSIZE(N), cudaMemcpyHostToDevice) );
}

void printMatrix(const double *mat, int m, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            printf("%f ", mat[j * m + i]);
        }
        printf("\n");
    }
}


/******************************************************************************
 * MAIN() function
 *****************************************************************************/

int main(int argc, char *argv[])
{   

	typedef MATRIX_DATA_TYPE value_type;

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
		// Define and initialize network
		network_t<value_type> alexnet;
		Layer_t<value_type> conv1; 	conv1.initConvLayer("conv1", 1, 20, 5, IMAGE_H, IMAGE_W);

		Layer_t<value_type> pool1; 	pool1.initPoolLayer("pool1", 2, 2, conv1);

		Layer_t<value_type> conv2; 	conv2.initConvLayer("conv2", conv1.outputs, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride, conv1.outputs * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
		Layer_t<value_type> pool2; 	pool2.initPoolLayer("pool2", 2, 2, conv2);

		Layer_t<value_type> fc1;	fc1.initFCLayer(	"fc1", (conv2.outputs*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 500);
		Layer_t<value_type> fc1act; fc1act.initLayer(	"fc1act", ACT_LAYER, fc1.outputs);

		Layer_t<value_type> fc2; 	fc2.initFCLayer(	"fc2", fc1act.outputs, 10);

		Layer_t<value_type> fc2act; fc2act.initLayer(	"fc2act", ACT_LAYER, fc2.outputs);
	
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
	
		// Try to load learned weights from file other wise start learning phase
		if (conv1.load() && conv2.load() && fc1.load() && fc2.load())
		{
			conv1.copyDataToDevice();
			conv2.copyDataToDevice();
			fc1.copyDataToDevice();
			fc2.copyDataToDevice();
			println("Weights from file loaded");
		}
		else{
			println("\n **** Learning started ****");
			std::clock_t    start;
			start = std::clock(); 
	
			// Learn all examples till convergence
			value_type imgData_h[N];
			value_type* image_data_d = NULL;
			checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(N)) );
			int num_iterations = 1;
			while(num_iterations--){ // TODO: Use a better convergence criteria
				for (int i=0; i<m; i++){
					if (DEBUG) print("\n\n\n\n\n");
					const value_type *training_example = train_data+i*N;
					value_type target = train_target[i];
					for (int ii = 0; ii < N; ii++)
					{
						imgData_h[ii] = training_example[ii] / value_type(255);
						if (DEBUG){
							print((imgData_h[ii]>0?"#":" ")<<" ");
							if (ii%IMAGE_W==IMAGE_W-1)
								println(" ");
						}
					}
					
					checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, MSIZE(N), cudaMemcpyHostToDevice) );
					value_type predicted = alexnet.learn_example(image_data_d, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act, target);
					if (DEBUG) getchar();
					else if (i%1000==0) print("."<<std::flush);
					//println("Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
				}
			}
			checkCudaErrors( cudaFree(image_data_d) );
			println("\n **** Learning completed ****");
			println("Learning Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			
			
			conv1.copyDataToHost();
			conv2.copyDataToHost();
			fc1.copyDataToHost();
			fc2.copyDataToHost();
			// Save the weights in a binary file
			if (conv1.save() && conv2.save() && fc1.save() && fc2.save())
				println("Weights Saved.");
		}
	
		// Testing Phase
		{
			println("\n **** Testing started ****");
			std::clock_t    start;
			start = std::clock(); 
			int correct = 0;
			int n = total_test_data/N;
			value_type* image_data_d = NULL;
			value_type imgData_h[N];
			checkCudaErrors( cudaMalloc(&image_data_d, MSIZE(N)) );	
			for (int i=0; i<n; i++){
				const value_type *test_example = testing_data+i*N;
				value_type target = testing_target[i];
				for (int ii = 0; ii < N; ii++)
				{
					imgData_h[ii] = test_example[ii] / value_type(255);
					if (DEBUG){
						print((imgData_h[ii]>0?"#":" ")<<" ");
						if (ii%IMAGE_W==IMAGE_W-1)
							println(" ");
					}
				}
				checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, MSIZE(N), cudaMemcpyHostToDevice) );
				value_type predicted = alexnet.predict_example(image_data_d, conv1, pool1, conv2, pool2, fc1, fc1act, fc2, fc2act);
				
				if (target == predicted){
					correct++;
				}
				if (!DEBUG && i%1000==0) print("."<<std::flush);
				// println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
			}
			checkCudaErrors( cudaFree(image_data_d) );
			println("\n **** Testing completed ****\n");
			println("Testing Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
			println("Correctly predicted "<<correct<<" examples out of "<<n);
		}
	}

	// Reset device and exit gracefully
	cudaDeviceReset();
	exit(EXIT_SUCCESS);        
}
