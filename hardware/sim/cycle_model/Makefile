# ==========================================================================
# Makefile for SMGP Cycle-Accurate SystemC Simulator
# ==========================================================================

CXX       ?= g++
SYSTEMC_HOME ?= /usr/local/systemc

# -- Source files ------------------------------------------------------------
SRCS      = sc_main.cpp smgp_sc_module.cpp
HEADERS   = smgp_sc_module.h

# -- Output -----------------------------------------------------------------
TARGET    = smgp_cycle_sim

# -- Compiler flags ---------------------------------------------------------
CXXFLAGS  = -I$(SYSTEMC_HOME)/include -std=c++14 -O2 -Wall -Wextra
LDFLAGS   = -L$(SYSTEMC_HOME)/lib -Wl,-rpath,$(SYSTEMC_HOME)/lib

# SystemC library (with or without -lpthread depending on platform)
UNAME_S   := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -lsystemc
else
    LDFLAGS += -lsystemc -lpthread
endif

# -- Object files -----------------------------------------------------------
OBJS      = $(SRCS:.cpp=.o)

# ===========================================================================
# Targets
# ===========================================================================

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Clean complete."

# -- Debug build ------------------------------------------------------------
debug: CXXFLAGS += -g -DSC_DEBUG
debug: clean $(TARGET)
	@echo "Debug build complete."

# -- Shared library build (for Python ctypes interface) ---------------------
lib: CXXFLAGS += -fPIC -DSHARED_LIB
lib: LDFLAGS  += -shared
lib: smgp_cycle_sim.so

smgp_cycle_sim.so: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Shared library build complete: $@"
