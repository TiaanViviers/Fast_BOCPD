# Fast BOCPD - Unified Build System
# ===================================

CC = gcc
CFLAGS = -std=c99 -O3 -march=native -fomit-frame-pointer -Wall -Wextra -fPIC
# Note: -ffast-math and -funroll-loops tested but reduced performance
INCLUDES = -I$(shell python3 -c "import numpy; print(numpy.get_include())")
LDFLAGS = -shared -lm

# Directories
SRC_DIR = fast_bocpd/_c
TEST_DIR = tests/c_tests
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
LIB_DIR = $(BUILD_DIR)/lib

# Source files
SRC_FILES = $(SRC_DIR)/gaussian_nig.c $(SRC_DIR)/student_t_ng.c \
            $(SRC_DIR)/student_t_ng_grid.c $(SRC_DIR)/poisson_gamma.c \
            $(SRC_DIR)/bernoulli_beta.c $(SRC_DIR)/binomial_beta.c \
            $(SRC_DIR)/gamma_gamma_fixed_shape.c \
            $(SRC_DIR)/hazard.c $(SRC_DIR)/bocpd_core.c

IMPL_SRCS = $(SRC_DIR)/gaussian_nig.c $(SRC_DIR)/hazard.c $(SRC_DIR)/bocpd_core.c \
            $(SRC_DIR)/student_t_ng.c $(SRC_DIR)/student_t_ng_grid.c \
            $(SRC_DIR)/poisson_gamma.c $(SRC_DIR)/bernoulli_beta.c \
            $(SRC_DIR)/binomial_beta.c $(SRC_DIR)/gamma_gamma_fixed_shape.c
IMPL_OBJS = $(OBJ_DIR)/gaussian_nig.o $(OBJ_DIR)/hazard.o $(OBJ_DIR)/bocpd_core.o \
            $(OBJ_DIR)/student_t_ng.o $(OBJ_DIR)/student_t_ng_grid.o \
            $(OBJ_DIR)/poisson_gamma.o $(OBJ_DIR)/bernoulli_beta.o \
            $(OBJ_DIR)/binomial_beta.o $(OBJ_DIR)/gamma_gamma_fixed_shape.o

TEST_SRCS = $(TEST_DIR)/test_gaussian_nig.c $(TEST_DIR)/test_student_t_ng.c \
            $(TEST_DIR)/test_student_t_ng_grid.c $(TEST_DIR)/test_poisson_gamma.c \
            $(TEST_DIR)/test_bernoulli_beta.c $(TEST_DIR)/test_binomial_beta.c \
            $(TEST_DIR)/test_gamma_gamma.c \
            $(TEST_DIR)/test_hazard.c $(TEST_DIR)/test_bocpd_core.c $(TEST_DIR)/test_runner.c
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SRCS))

# Targets
LIB_TARGET = $(LIB_DIR)/libbocpd.so
TEST_RUNNER = $(BUILD_DIR)/test_runner

.PHONY: all lib test test-c test-python test-sanitizers test-valgrind benchmark clean help

# Default target
all: help

# Build shared library (for development without pip install)
lib: $(LIB_TARGET)

$(LIB_TARGET): $(IMPL_OBJS) | $(LIB_DIR)
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	@echo "✓ Shared library built: $(LIB_TARGET)"

# Build C test runner
$(TEST_RUNNER): $(IMPL_OBJS) $(TEST_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm
	@echo "✓ Test runner built: $(TEST_RUNNER)"

# Compile implementation source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test source files
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Create build directories
$(BUILD_DIR) $(OBJ_DIR) $(LIB_DIR):
	mkdir -p $@

# Run C tests
test-c: $(TEST_RUNNER)
	@echo ""
	@LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(TEST_RUNNER)

# Run Python tests
test-python:
	@echo ""
	@echo "Running Python tests..."
	@pytest

# Run all tests (C + Python)
test: test-c test-python

# Run tests with AddressSanitizer + UndefinedBehaviorSanitizer
test-sanitizers:
	@echo ""
	@echo "========================================="
	@echo "Running with ASan + UBSan"
	@echo "========================================="
	@$(MAKE) clean > /dev/null 2>&1
	@CFLAGS="-std=c99 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra" $(MAKE) $(TEST_RUNNER) > /dev/null 2>&1
	@echo ""
	@$(TEST_RUNNER)
	@echo ""
	@echo "✓ Sanitizer tests passed!"
	@$(MAKE) clean > /dev/null 2>&1

# Run tests with Valgrind
test-valgrind: $(TEST_RUNNER)
	@echo ""
	@echo "========================================="
	@echo "Running with Valgrind"
	@echo "========================================="
	@echo ""
	@valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 $(TEST_RUNNER)
	@echo ""
	@echo "✓ Valgrind tests passed!"

# Run benchmarks
benchmark: 
	cd benchmarks && ./benchmark.sh && cd ..

# Clean all build artifacts
clean:
	rm -rf $(BUILD_DIR)
	find . -type f -name "*.o" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.npy" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "✓ Clean complete"

# Help message
help:
	@echo "Fast BOCPD Build System"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  make lib              Build shared library (for development)"
	@echo "  make test             Run all tests (C + Python)"
	@echo "  make test-c           Run C unit tests only"
	@echo "  make test-python      Run Python tests only"
	@echo "  make test-sanitizers  Run C tests with ASan/UBSan"
	@echo "  make test-valgrind    Run C tests with Valgrind"
	@echo "  make benchmark        Run benchmarks"
	@echo "  make clean            Remove all build artifacts"
	@echo "  make help             Show this help message"
	@echo ""
	@echo "Development workflow:"
	@echo "  1. Edit C code in fast_bocpd/_c/"
	@echo "  2. make test      # Verify changes"
	@echo "  3. pip install -e .  # For Python integration"
	@echo ""
	@echo "Note: 'pip install' automatically compiles C code."
	@echo "      'make lib' is only needed for manual testing."
