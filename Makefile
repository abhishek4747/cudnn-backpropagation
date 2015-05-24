CUDA_PATH=/usr/local/cuda-6.5
CUDNN_PATH=/home/cse/dual/cs5110272/Desktop/cudnn/cudnn-6.5-linux-x64-v2

CC=g++
CFLAGS=-I. -I$(CUDA_PATH)/include -I$(CUDNN_PATH) -IFreeImage/include -IUtilNPP
LIBS=-lcudart -lnppi -lnppc -lcublas -lcudnn -lfreeimage -lm -lstdc++
LFLAGS=-L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH) -L./FreeImage/lib/linux/x86_64 $(LIBS)

OBJ = mnistCUDNN.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mnistCUDNN: $(OBJ)
	gcc -o $@ $^ $(LFLAGS)

clean:
	rm *.o ./mnistCUDNN
