CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -I../../fast_bocpd/_c
LDFLAGS = -lm
C_SRC_DIR = ../../fast_bocpd/_c

# Source files from the main implementation
IMPL_SRCS = $(C_SRC_DIR)/gaussian_nig.c $(C_SRC_DIR)/hazard.c $(C_SRC_DIR)/bocpd_core.c
IMPL_OBJS = gaussian_nig.o hazard.o bocpd_core.o

# Test files
TEST_SRCS = test_gaussian_nig.c test_hazard.c test_bocpd_core.c test_runner.c
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Targets
TEST_RUNNER = test_runner
ALL_OBJS = $(IMPL_OBJS) $(TEST_OBJS)

.PHONY: all clean test help

all: test

# Build test runner
$(TEST_RUNNER): $(IMPL_OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile implementation source files
%.o: $(C_SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test source files
%.o: %.c test_utils.h
	$(CC) $(CFLAGS) -c $< -o $@

# Run tests
test: $(TEST_RUNNER)
	@echo ""
	@./$(TEST_RUNNER)

# Clean build artifacts
clean:
	rm -f $(ALL_OBJS) $(TEST_RUNNER)

# Help
help:
	@echo "C Unit Tests Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make test   - Build and run all C unit tests (default)"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"
	@echo ""
	@echo "Test Files:"
	@echo "  test_gaussian_nig.c  - Tests for GaussianNIG observation model"
	@echo "  test_hazard.c        - Tests for hazard functions"
	@echo "  test_bocpd_core.c    - Tests for main BOCPD algorithm"
	@echo "  test_runner.c        - Main test runner"
