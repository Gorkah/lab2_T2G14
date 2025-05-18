# Compiler and flags
CC = mpicc
CFLAGS = -O3 -Wall -lm

# Target executable
TARGET = montecarlo

# Source file
SRC = montecarlo.c

# Default target
all: $(TARGET)

# Compile the program
$(TARGET): $(SRC)
    $(CC) $(CFLAGS) -o $@ $<

# Clean up generated files
clean:
    rm -f $(TARGET)