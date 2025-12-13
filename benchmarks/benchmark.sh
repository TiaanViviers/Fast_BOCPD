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

benchmark_Fbocpd() {
    echo "Running Fast-BOCPD benchmarks"
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution gaussian
    python3 benchmark_fast_bocpd.py --distribution student_t_fixed
    python3 benchmark_fast_bocpd.py --distribution student_t_grid
    python3 benchmark_fast_bocpd.py --distribution bernoulli
    python3 benchmark_fast_bocpd.py --distribution binomial
    python3 benchmark_fast_bocpd.py --distribution poisson
    python3 benchmark_fast_bocpd.py --distribution gamma
    cd ..
}

benchmark_competitors() {
    echo "Running competitor benchmarks"
    cd scripts
        python3 benchmark_competitors.py --lib dtolpin
        python3 benchmark_competitors.py --lib ruptures
        python3 benchmark_competitors.py --lib hildensia
        python3 benchmark_competitors.py --lib promised-ai

    cd ..
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


if [ $# == 0 ]; then
    benchmark_Fbocpd
    benchmark_competitors
elif [ "$1" == "Fbocpd" ]; then
    benchmark_Fbocpd
elif [ "$1" == "competitors" ]; then
    benchmark_competitors
elif [ "$1" == "gaussian" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution gaussian
    cd ..
elif [ "$1" == "student_t_fixed" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution student_t_fixed
    cd ..
elif [ "$1" == "student_t_grid" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution student_t_grid
    cd ..
elif [ "$1" == "bernoulli" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution bernoulli
    cd ..
elif [ "$1" == "binomial" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution binomial
    cd ..
elif [ "$1" == "poisson" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution poisson
    cd ..
elif [ "$1" == "gamma" ]; then
    cd scripts
    python3 benchmark_fast_bocpd.py --distribution gamma
    cd ..
else
    echo "Please provide a valid argument:"
    echo "  <no arguments>        Run all benchmarks"
    echo "  Fbocpd                Run Fast-BOCPD benchmarks"
    echo "  competitors           Run competitor benchmarks"
    echo "  <distribution_name>   Run benchmark for specific distribution"
fi
