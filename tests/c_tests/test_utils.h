#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <math.h>

/* Color codes for terminal output */
#define COLOR_GREEN "\033[0;32m"
#define COLOR_RED "\033[0;31m"
#define COLOR_RESET "\033[0m"

/* Test assertion macros */
#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            printf(COLOR_RED "  ✗ FAIL: " COLOR_RESET "%s (line %d)\n", #expr, __LINE__); \
            return 1; \
        } \
    } while(0)

#define ASSERT_FALSE(expr) \
    do { \
        if (expr) { \
            printf(COLOR_RED "  ✗ FAIL: " COLOR_RESET "!(%s) (line %d)\n", #expr, __LINE__); \
            return 1; \
        } \
    } while(0)

#define ASSERT_CLOSE(a, b, tol) \
    do { \
        double _a = (a); \
        double _b = (b); \
        if (fabs(_a - _b) > (tol)) { \
            printf(COLOR_RED "  ✗ FAIL: " COLOR_RESET "%s ≈ %s (line %d)\n", #a, #b, __LINE__); \
            printf("    Expected: %.10f, Got: %.10f, Diff: %.10e\n", _b, _a, fabs(_a - _b)); \
            return 1; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            printf(COLOR_RED "  ✗ FAIL: " COLOR_RESET "%s == %s (line %d)\n", #a, #b, __LINE__); \
            return 1; \
        } \
    } while(0)

/* Test suite macros */
#define TEST_SUITE(name) \
    printf("\n" COLOR_GREEN "Testing %s" COLOR_RESET "\n", name)

#define TEST_PASS(name) \
    do { \
        printf(COLOR_GREEN "  ✓ " COLOR_RESET "%s\n", name); \
    } while(0)

#endif /* TEST_UTILS_H */
