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

 * The sample can work in single, double, half precision, but it
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

#include "ImageIO.h"
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
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
 * HELPER FUNCTIONS for classes
 *****************************************************************************/

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
	sFilename = (std::string("datav5/") + std::string(fname));
}

template <typename value_type> 
void printHostVector(std::string str, int size, value_type* vec){
	print(str);
	for (int i = 0; i < minn(size,400); i++)
	{
		print(std::setprecision(2) << vec[i] << " ");
	}
	println(" "); 
}

template <typename value_type>
void printDeviceVector(std::string str, int size, value_type* vec_d)
{
	value_type *vec;
	vec = new value_type[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
	printHostVector(str, size, vec);
	delete [] vec;
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};

template <> 
struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
	template <class T>
	value_type operator()(T x) {return value_type(x);}
	value_type operator()(half1 x) {return value_type(cpu_half2float(x));}
};

template <>
class Convert<half1>
{
public:
	template <class T>
	half1 operator()(T x) {return cpu_float2half_rn (T(x));} 
	half1 operator()(half1 x) {return x;}
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

	int size_b = size*sizeof(value_type);
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
	cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
	Convert<real_type> toReal;
	std::cout.precision(7);
	std::cout.setf( std::ios::fixed, std:: ios::floatfield );
	for (int i = 0; i < size; i++)
	{
		print(toReal(vec[i]) << " ");
	}
	println(" ");
	delete [] vec;
}

/******************************************************************************
 * Where to perform fp16 calculation (most probably deprecated)
 *****************************************************************************/

typedef enum {
		FP16_HOST  = 0, 
		FP16_CUDA  = 1,
		FP16_CUDNN = 2
 } fp16Import_t;


/******************************************************************************
 * Layer_t struct : contains information about layers
 *****************************************************************************/
template <class value_type>
struct Layer_t
{
	fp16Import_t fp16Import;
	int inputs;
	int outputs;
	// linear dimension (i.e. size is kernel_dim * kernel_dim)
	int kernel_dim;
	value_type *data_h, *data_d;
	value_type *bias_h, *bias_d;
	value_type *output_h, *output_d;
	value_type *del_h, *del_d;

	Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
				inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){};
	Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
			const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
				  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
	{
		fp16Import = _fp16Import;
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
		readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
						&data_h, &data_d);
		readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
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

	~Layer_t()
	{
		if (data_h != NULL) delete [] data_h;
		if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
		if (bias_h != NULL) delete [] bias_h;
		if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
	}

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
private:
	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
	}
};

template <>
void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
{
	*data_h = new half1[size];
	int size_b = size*sizeof(half1);
	checkCudaErrors( cudaMalloc(data_d, size_b) );    
	float *data_tmp_h, *data_tmp_d;

	switch(fp16Import)
	{
		case FP16_HOST :
		{
			readBinaryFile<half1>(fname, size, *data_h);
			checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b,
								cudaMemcpyHostToDevice) );
			break;
		}
		case FP16_CUDA :
		{
			readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);

			gpu_float2half_rn<float>(size, data_tmp_d, *data_d);

			delete [] data_tmp_h;
			checkCudaErrors( cudaFree(data_tmp_d) );
			break;
		}
		case FP16_CUDNN :
		{
			readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
			delete [] data_tmp_h;
			cudnnHandle_t cudnnHandle;
			cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
			checkCUDNN( cudnnCreate(&cudnnHandle) );
			checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
			checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
			checkCUDNN( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
												CUDNN_DATA_FLOAT,
												1, size,
												1, 1,
												size, 1, 1, 1) );
			checkCUDNN( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
												CUDNN_DATA_HALF,
												1, size,
												1, 1,
												size, 1, 1, 1) );
			float alpha = 1.0f;
			float beta = 0.0f;
			checkCUDNN( cudnnTransformTensor(cudnnHandle, &alpha,
											 srcTensorDesc,
											 data_tmp_d, &beta,
											 dstTensorDesc,
											 *data_d) );
			checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
			checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
			checkCUDNN( cudnnDestroy(cudnnHandle) );
			checkCudaErrors( cudaFree(data_tmp_d) );
			break;
		}
	}
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
			case 2 : dataType = CUDNN_DATA_HALF; break;
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
		checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
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

	void fullyConnectedForward(const Layer_t<value_type>& ip,
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

		scaling_type alpha = scaling_type(1), beta = scaling_type(1);
		// place bias into dstData
		checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		
		gemv(cublasHandle, dim_x, dim_y, alpha,
				ip.data_d, srcData, beta,*dstData);

		h = 1; w = 1; c = dim_y;
	}

	void convoluteForward(const Layer_t<value_type>& conv,
						  int& n, int& c, int& h, int& w,
						  value_type* srcData, value_type** dstData)
	{
		cudnnConvolutionFwdAlgo_t algo;

		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims] = {n,c,h,w};
		const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
										conv.kernel_dim, conv.kernel_dim};
									   
		checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
											  dataType,
											  CUDNN_TENSOR_NCHW,
											  tensorDims,
											  filterDimA) );
 
		const int convDims = 2;
		int padA[convDims] = {0,0};
		int filterStrideA[convDims] = {1,1};
		int upscaleA[convDims] = {1,1};
		cudnnDataType_t  convDataType = dataType;
		if (dataType == CUDNN_DATA_HALF) {
			convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
		}
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

		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

		if (convAlgorithm < 0)
		{
			// Choose the best according to the preference
			println("Testing cudnnGetConvolutionForwardAlgorithm ...");
			checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
													srcTensorDesc,
													filterDesc,
													convDesc,
													dstTensorDesc,
													CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
													0,
													&algo
													) );
			println("Fastest algorithm is Algo " << algo);
			convAlgorithm = algo;
			// New way of finding the fastest config
			// Setup for findFastest call
			println("Testing cudnnFindConvolutionForwardAlgorithm ...");
			int requestedAlgoCount = 5; 
			int returnedAlgoCount[1];
			cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(
				sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
			checkCUDNN(cudnnFindConvolutionForwardAlgorithm( cudnnHandle, 
													 srcTensorDesc,
													 filterDesc,
													 convDesc,
													 dstTensorDesc,
													 requestedAlgoCount,
													 returnedAlgoCount,
													 results
												   ) );
			for(int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex){
				printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", 
					cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, 
					(unsigned long long)results[algoIndex].memory);
			}
			free(results);
		}
		else
		{
			algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
			if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
			{
				//println("Using FFT for convolution");
			}
		}

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
		scaling_type alpha = scaling_type(1);
		scaling_type beta  = scaling_type(0);
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
		const int poolDims = 2;
		int windowDimA[poolDims] = {2,2};
		int paddingA[poolDims] = {0,0};
		int strideA[poolDims] = {2,2};
		checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
												CUDNN_POOLING_MAX,
												CUDNN_PROPAGATE_NAN,
												poolDims,
												windowDimA,
												paddingA,
												strideA ) );

		setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims] = {n,c,h,w};
		checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
													srcTensorDesc,
													tensorDims,
													tensorOuputDimA) );
		n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

		setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
	 
		resize(n*c*h*w, dstData);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
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
										  *dstData) );
	}
	void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
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

	void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	{
		checkCUDNN( cudnnSetActivationDescriptor(activDesc,
												CUDNN_ACTIVATION_SIGMOID,
												CUDNN_PROPAGATE_NAN,
												0.0) );
	
		resize(n*c*h*w, dstData);

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
											*dstData) );    
	}

	void fullyConnectedBackward(const Layer_t<value_type>& current_layer, const value_type* last_input){
		int dim_x = current_layer.inputs;
		int dim_y = current_layer.outputs;
		int dim_z = 1;
		value_type* dstData = NULL;
		resize(dim_x*dim_y, &dstData);

		value_type alpha = value_type(1), beta = value_type(0);
		// checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, ip.outputs*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		//if (DEBUG) printDeviceVector("last_input: \n", current_layer.inputs, last_input);
		//if (DEBUG) printDeviceVector("del_W: \n", current_layer.outputs, current_layer.del_d);
		
		checkCudaErrors( CUBLAS_GEMM(cublasHandle, 
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
		
		checkCudaErrors( CUBLAS_GEAM(cublasHandle,
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
		checkCudaErrors( CUBLAS_GEAM(cublasHandle,
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

	void activationBackward(int n, int c, int h, int w, value_type* srcData, value_type* dstData, value_type *srcDiffData, value_type **dstDiffData)
	{
		checkCUDNN( cudnnSetActivationDescriptor(activDesc,
												CUDNN_ACTIVATION_SIGMOID,
												CUDNN_PROPAGATE_NAN,
												0.0) );
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
											activDesc, // RELU
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

	int classify_example(const char* fname, const Layer_t<value_type>& conv1,
						  const Layer_t<value_type>& conv2,
						  const Layer_t<value_type>& ip1,
						  const Layer_t<value_type>& ip2)
	{
		int n,c,h,w;
		value_type *srcData = NULL, *dstData = NULL;
		value_type imgData_h[IMAGE_H*IMAGE_W];

		readImage(fname, imgData_h);

		println("Performing forward propagation ...");

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
		lrnForward(n, c, h, w, srcData, &dstData);

		fullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);
		softmaxForward(n, c, h, w, srcData, &dstData);

		const int max_digits = 10;
		// Take care of half precision
		Convert<scaling_type> toReal;
		value_type result[max_digits];
		checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
		int id = 0;
		for (int i = 1; i < max_digits; i++)
		{
			if (toReal(result[id]) < toReal(result[i])) id = i;
		}

		println("Resulting weights from Softmax:");
		printDeviceVector(n*c*h*w, dstData);

		checkCudaErrors( cudaFree(srcData) );
		checkCudaErrors( cudaFree(dstData) );
		return id;
	}

	int predictExample(const value_type** image_data, value_type target, const Layer_t<value_type>& input, const Layer_t<value_type>& hidden){
		
		value_type *image_data_d = NULL;
		value_type imgData_h[IMAGE_H*IMAGE_W];

		// Plot to console and normalize image to be in range [0,1]
		for (int i = 0; i < N; i++)
		{
			imgData_h[i] = (*image_data)[i] / value_type(255);
			#if 0
			print((imgData_h[i]>0?"#":" ")<<" ");
			if (i%IMAGE_W==IMAGE_W-1)
				println(" ");
			#endif
		}
		resize(IMAGE_H*IMAGE_W, &image_data_d);
		checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyHostToDevice) );
		int id = predictExampleDevice(image_data_d, target, input, hidden);
		checkCudaErrors( cudaFree(image_data_d) );
		return id;
	}

	int predictExampleDevice(const value_type* image_data_d, value_type target, const Layer_t<value_type>& input, const Layer_t<value_type>& hidden){
		int n,c,h,w;
		value_type *srcData = NULL, *dstData = NULL;
		if (DEBUG) println("\nTarget: "<<target);

		// Setup Variables for forward propagation
		//checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) ); 
		resize(IMAGE_H*IMAGE_W, &srcData);
		checkCudaErrors( cudaMemcpy(srcData, image_data_d,  IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyDeviceToDevice) );
		n = c = 1; h = IMAGE_H; w = IMAGE_W;
		// Perform Forward propagation
		if (DEBUG) println("Performing forward propagation ...");
 
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

	int learnExample(const value_type** image_data, value_type target, const Layer_t<value_type>& input,
						  const Layer_t<value_type>& hidden)
	{
	   
		value_type *image_data_d = NULL;
		value_type imgData_h[IMAGE_H*IMAGE_W];

		// Plot to console and normalize image to be in range [0,1]
		for (int i = 0; i < N; i++)
		{
			imgData_h[i] = (*image_data)[i] / value_type(255);
			#if 0
			print((imgData_h[i]>0?"#":" ")<<" ");
			if (i%IMAGE_W==IMAGE_W-1)
				println(" ")
			#endif
		}
		resize(IMAGE_H*IMAGE_W, &image_data_d);
		checkCudaErrors( cudaMemcpy(image_data_d, imgData_h, IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyHostToDevice) );
		int id = predictExampleDevice(image_data_d, target, input, hidden);
		if (DEBUG) println("Prediction: "<<id);
		
		
		value_type *srcDiffData = NULL, *dstDiffData = NULL, *targetData = NULL;
		int n, c, h, w;

		// Perform backward propagation
		if (DEBUG) println("\nPerforming backward propagation ...");
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
	
	void getBackPropData(const Layer_t<value_type>& layer, const Layer_t<value_type>& next_layer, value_type target, value_type* dstDiffData, 
			value_type** targetData, value_type** srcDiffData, 
			bool last_layer)
	{
		const int max_digits = layer.outputs;
		value_type srcData[max_digits];
		checkCudaErrors( cudaMemcpy(srcData, layer.output_d, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
		resize(max_digits, srcDiffData);
		resize(max_digits, targetData);
		if (last_layer){
			// resize(max_digits, &dstDiffData);
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
			checkCudaErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_N,
									  next_layer.inputs, next_layer.outputs,
									  &alpha,
									  next_layer.data_d, next_layer.inputs,
									  dstDiffData, 1,
									  &beta,
									  *srcDiffData, 1) );
		}
		if (DEBUG) printDeviceVector("\tdstDiffData: \n", max_digits, dstDiffData);
		if (DEBUG) printDeviceVector("\ttargetData: \n", max_digits, *targetData);
		if (DEBUG) printDeviceVector("\tsrcDiffData: \n", max_digits, *srcDiffData);
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

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
						  int& n, int& c, int& h, int& w,
						  half1* srcData, half1** dstData)
{
	c = c*h*w; h = 1; w = 1;
	network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
	c = ip.outputs;
}
#endif

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
bool loadWeights(const char* filename, size_t size, value_type* matrix){
	std::ifstream myfile(filename, std::ios::in | std::ios::binary);
	if (myfile.is_open()){
		myfile.read((char*)matrix, size*sizeof(value_type));
	}else{
		println("Error reading file "<<filename);
		return false;
	}
}

template <typename value_type>
bool saveWeights(const char* filename, size_t size, value_type* matrix){
	std::ofstream myfile(filename, std::ios::out | std::ios::binary);
	if (myfile.is_open()){
		myfile.write((char*)matrix, size*sizeof(value_type));
	}else{
		println("Error saving file "<<filename);
		return false;
	}
}

/******************************************************************************
 * MAIN() function
 *****************************************************************************/

int main(int argc, char *argv[])
{   
	// Define the precision you want to use: full (float), double (double)
	typedef MATRIX_DATA_TYPE value_type;

	// Print Usage if help is in the arguments
	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		displayUsage();
		exit(EXIT_WAIVED); 
	}

	// Print Library and Device stats
	int version = (int)cudnnGetVersion();
	println("\n\n\n");
	printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
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
	
	const char *conv1_bin = "conv1.bin";
	const char *conv1_bias_bin = "conv1.bias.bin";
	const char *conv2_bin = "conv2.bin";
	const char *conv2_bias_bin = "conv2.bias.bin";
	const char *ip1_bin = "ip1.bin";
	const char *ip1_bias_bin = "ip1.bias.bin";
	const char *ip2_bin = "ip2.bin";
	const char *ip2_bias_bin = "ip2.bias.bin";

	network_t<value_type> mnist;
	Layer_t<value_type> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
	Layer_t<value_type> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
	Layer_t<value_type>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
	Layer_t<value_type>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);

	// Define and initialize network
	Layer_t<value_type> input(N,100);
	Layer_t<value_type> hidden(100,10);

	value_type *train_data, *testing_data;
	value_type *train_target, *testing_target;

	// Read training data
	value_type *training_data;
	value_type *training_target;
	int total_train_data, total_test_data;
	mnist.loadData(&training_data, &testing_data, &training_target, &testing_target, total_train_data, total_test_data);
	println("\n\nData Loaded. Training examples:"<<total_train_data/N<<" Testing examples:"<<total_test_data/N);

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
	println("Training Examples shuffled.");

	// Free some variables
	delete [] training_data;
	delete [] training_target;

	// Try to load learned weights from file other wise start learning phase
	if (loadWeights("input_data.bin", input.inputs*input.outputs, input.data_h) &&
		loadWeights("input_bias.bin", input.outputs, input.bias_h) &&
		loadWeights("hidden_data.bin", hidden.inputs*hidden.outputs, hidden.data_h) &&
		loadWeights("hidden_bias.bin", hidden.outputs, hidden.bias_h))
	{
		input.copyDataToDevice();
		hidden.copyDataToDevice();
		println("Weights from file loaded");
	}
	else{
		println("\n **** Learning started ****");
		std::clock_t    start;
		start = std::clock(); 

		// Learn all examples till convergence
		int num_iterations = 5;
		while(num_iterations--){ // Use a better convergence criteria
			for (int i=0; i<m; i++){
				if (DEBUG) print("\n\n\n\n\n");
				const value_type *training_example = train_data+i*N;
				value_type target = train_target[i];
				value_type predicted = mnist.learnExample(&training_example, target, input, hidden);
				if (DEBUG) getchar();
				else if (i%1000==0) print("."<<std::flush);
				//println("Example "<<i<<" learned. "<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
			}
		}
		println("\n **** Learning completed ****");
		println("Learning Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
		
		input.copyDataToHost();
		hidden.copyDataToHost();
		// Save the weights in a binary file
		saveWeights("input_data.bin", input.inputs*input.outputs, input.data_h);
		saveWeights("input_bias.bin", input.outputs, input.bias_h);
		saveWeights("hidden_data.bin", hidden.inputs*hidden.outputs, hidden.data_h);
		saveWeights("hidden_bias.bin", hidden.outputs, hidden.bias_h);
	}

	// Testing Phase
	{
		println("\n **** Testing started ****");
		std::clock_t    start;
		start = std::clock(); 
		int correct = 0;
		int n = total_test_data/N;
		for (int i=0; i<n; i++){
			const value_type *test_example = testing_data+i*N;
			value_type target = testing_target[i];
			value_type predicted = mnist.predictExample(&test_example, target, input, hidden);
			if (target == predicted){
				correct++;
			}
			if (!DEBUG && i%1000==0) print("."<<std::flush);
			//println("Example: "<<i<<"\tTarget: "<<target<<"\tPredicted: "<<predicted);
		}
		println("\n **** Testing completed ****\n");
		println("Testing Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " second");
		println("Correctly predicted "<<correct<<" examples out of "<<n);
	}

	// Reset device and exit gracefully
	cudaDeviceReset();
	exit(EXIT_SUCCESS);        
	
}
