# Compiler and linker settings
NVCC = nvcc
NVCCFLAGS = -arch=sm_52 -O3 -Xcompiler -Wall -rdc=true
LDFLAGS = -lcurand 
INCLUDES = -I/home/dcq/Dependence/eigen-3.4.0
# Target executable name
TARGET = kickModel

# Object files to be created
OBJS = main.o Interract.o Cell.o Box.o InputOutput.o Particle.o var.o System.o PullMethod.o

# Compile and link
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
# $(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)
