import sys
import numpy as np


def main():
    dim, distr, n, seg_len = parse_args(sys.argv[1:])
    if dim == 'uv':
        data = generate_univariate(distr, n, seg_len)
    elif dim == 'mv':
        pass
        #data = generate_multivariate(distr, n)

    write_data(data, dim, distr, n, seg_len)


def generate_univariate(distr, n, seg_len):
    data = []
    num_segments = n // seg_len
    
    if distr == 'gaussian':
        data = generate_gaussian(n, seg_len, num_segments)
    return data


########################## Data Generation Functions ###########################
def generate_gaussian(n, seg_len, num_segments):
    data = []
    mu = 0
    
    for _ in range(num_segments):
        mu = (mu+5) if mu <= 40 else 0
        sigma = np.random.uniform(0.5, 2.5)
        segment = np.random.normal(loc=mu, scale=sigma, size=seg_len)
        data.extend(segment)
        
    # Handle any remaining samples if n is not a multiple of seg_len
    remaining = n % seg_len
    if remaining > 0:
        mu = (mu+5) if mu <= 40 else 0
        sigma = np.random.uniform(0.5, 2.5)
        segment = np.random.normal(loc=mu, scale=sigma, size=remaining)
        data.extend(segment)
        
    return np.array(data)
    
    
########################### Data Writing Function ##############################
def write_data(data, dim, distr, n, seg_len):
    data_dir = "../data/"
    filename = f"data_{dim}_{distr}_n{n}_seg{seg_len}.npy"
    np.save(data_dir + filename, data)
    print(f"Data saved to {data_dir + filename}")


#################### Argument Parsing ####################
def parse_args(args):
    if len(args) != 4:
        print("Usage: python generate_data.py\n" + 
              "\t<dim> dimension of the data (uv-univariate, mv-multivariate)\n" +
              "\t<distr> distribution (gaussian)\n" +
              "\t<n> number of samples\n" +
              "\t<seg_len> length of each segment")
        sys.exit(1)
        
    dim = args[0]
    if dim not in ['uv', 'mv']:
        print("Error: dimension must be 'uv' or 'mv'")
        sys.exit(1)
        
    distr = args[1]
    if distr not in ['gaussian']:
        print("Error: distribution must be 'gaussian'")
        sys.exit(1)

    n = int(args[2])
    if n <= 0:
        print("Error: number of samples must be positive")
        sys.exit(1)

    seg_len = int(args[3])
    if seg_len <= 0:
        print("Error: segment length must be positive")
        sys.exit(1)

    return dim, distr, n, seg_len


if __name__ == "__main__":
    main()