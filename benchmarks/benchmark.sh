# Fast-BOCPD Benchmark Runner
# Automatically generates data if missing and runs benchmarks

test_data_exists() {
    if [ ! -f "data/data_uv_$1_n$2_seg150.npy" ]; then
        echo "Data file for $1 n=$2 not found. Generating..."
        cd scripts
        python3 generate_data.py uv $1 $2 150
        cd ..
    fi
}

# Check if data files exist for gaussian
echo "Checking data files..."
test_data_exists gaussian 1000
test_data_exists gaussian 10000
test_data_exists gaussian 100000

# Run benchmarks for univariate gaussian
echo ""
echo "Running benchmarks..."
cd scripts
python3 benchmark_uv_gaussian.py
cd ..