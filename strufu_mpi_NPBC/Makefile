HOST=$(shell hostname)

ifneq (,$(findstring Twister,$(HOST)))
CXX       = mpicxx
HDF5_PATH = /opt/local
SZIP_PATH = /opt/local
CCFLAGS = -O3 -g -Wall
LDFLAGS = -O3 -g -Wall
CCFLAGS += -I$(HDF5_PATH)/include -I$(SZIP_PATH)
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz -L$(SZIP_PATH) -lsz
endif

# Gadi or SuperMUC
ifneq (,$(findstring login,$(HOST)))
CXX       = mpiCC
HDF5_PATH = $(HDF5_BASE)
SZIP_PATH = $(SZIP_BASE)
CCFLAGS = -O3
LDFLAGS = -O3
CCFLAGS += -I$(HDF5_PATH)/include -I$(SZIP_PATH)
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz -L$(SZIP_PATH)
endif

BIN = strufu_mpi

all : $(BIN)

clean : 
	rm -f *.o *~ $(BIN)

$(BIN) : $(BIN).o
	$(CXX) -o $(BIN) $(BIN).o $(LDFLAGS)

$(BIN).o : $(BIN).cpp
	$(CXX) -c $(BIN).cpp $(CCFLAGS)
