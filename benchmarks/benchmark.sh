# Fast-BOCPD Benchmark Runner
# Automatically generates data if missing and runs benchmarks

test_data_exists() {
    if [ ! -f "data/data_$1_n$2_seg150.npy" ]; then
        echo "Data file for $1 n=$2 not found. Generating..."
        cd scripts
        python3 generate_data.py $1 $2 150
        cd ..
    fi
}

# Check if data files exist for gaussian
echo "Checking data files..."
test_data_exists gaussian 1000
test_data_exists gaussian 10000
test_data_exists gaussian 100000
# Fixed df student-t
test_data_exists student_t_fixed 1000
test_data_exists student_t_fixed 10000
test_data_exists student_t_fixed 100000
# Student-t (grid)
test_data_exists student_t_grid 1000
test_data_exists student_t_grid 10000
test_data_exists student_t_grid 100000
# Bernoulli
test_data_exists bernoulli 1000
test_data_exists bernoulli 10000
test_data_exists bernoulli 100000
# Binomial
test_data_exists binomial 1000
test_data_exists binomial 10000
test_data_exists binomial 100000
# Poisson
test_data_exists poisson 1000
test_data_exists poisson 10000
test_data_exists poisson 100000
# Gamma
test_data_exists gamma 1000
test_data_exists gamma 10000
test_data_exists gamma 100000

# Run benchmarks for univariate gaussian
echo ""
echo "Running benchmarks..."
cd scripts
#python3 benchmark_uv_gaussian.py
cd ..