CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -fPIC
LDFLAGS = -lm

SRCS = gaussian_nig.c hazard.c bocpd_core.c
OBJS = $(SRCS:.c=.o)
LIB_TARGET = libbocpd.so

.PHONY: all clean lib

# Default: build shared library for development
all: lib

# Shared library (for development/testing without pip install)
$(LIB_TARGET): $(OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

lib: $(LIB_TARGET)

clean:
	rm -f $(OBJS) $(LIB_TARGET)

help:
	@echo "Makefile for Fast BOCPD C library"
	@echo ""
	@echo "Targets:"
	@echo "  make lib    - Build shared library for development (default)"
	@echo "  make clean  - Remove build artifacts"
	@echo ""
	@echo "Note: For package installation, use 'pip install -e .' in the root directory."
	@echo "      For C unit tests, use 'make test' in the tests/c_tests/ directory."

