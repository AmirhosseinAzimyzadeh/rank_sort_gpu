using CUDA
using BenchmarkTools

CUDA.versioninfo()
@show CUDA.functional()

cu_device = CUDA.device()
max_threads = CUDA.attribute(cu_device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

# Generate an array in random order and return the array and its sorted version
# Used for testing purposes
function generateArray(n::Int64)
  a = rand(n)
  return (a, sort(a))
end

# Check the validity of sorted array
function checkSorted(a::Array{Float64, 1}, sorted::Array{Float64, 1})
  return a == sorted
end

function sequentialRankSort(a::Array{Float64, 1})
  n = length(a)
  b = similar(a)
  for i in 1:n
    idx = 0
    @inbounds for j in 1:n
      a[i] > a[j] && (idx += 1)
    end
    b[idx+1] = a[i]
  end
  return b
end


function cuArrayRankSort(a::Array{Float64, 1})
  n = length(a)
  a_gpu = CuArray(a)
  sorted_gpu = similar(a_gpu)
  temp_gpu::CuArray{Int64, 1} = CuArray(zeros(UInt8, n))

  @inbounds for i in 1:n
    CUDA.@allowscalar temp_gpu[1:n] .= a_gpu[i] .> a_gpu[1:n]
    # sum of 1s in temp_gpu
    idx = CUDA.reduce(+, temp_gpu)
    CUDA.@allowscalar sorted_gpu[idx+1] = a_gpu[i]
  end
  return Array(sorted_gpu)
end


function rankSortKernel!(
  a, # original array
  sorted, # to be stored in sorted
  chunk_size, # chuck size assigned to each thread
  n # size of the array
)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  # Min and Max index of the chunk
  min_idx = (i - 1) * chunk_size + 1
  max_idx = min_idx + chunk_size - 1
  max_idx > n && (max_idx = n)

  for i in min_idx:max_idx
    idx = 0
    @inbounds for j in 1:n
      a[i] > a[j] && (idx += 1)
    end
    sorted[idx+1] = a[i]
  end
  return nothing
end

function cuNativeRankSort(a::Array{Float64, 1}, chunck_size::Int64)
  n = length(a)
  a_gpu = CuArray(a)
  sorted_gpu = similar(a_gpu)
  number_of_blocks = ceil(Int, n / chunck_size)

  @cuda blocks=number_of_blocks threads=max_threads rankSortKernel!(a_gpu, sorted_gpu, chunck_size, n)

  return Array(sorted_gpu)
end



# ---- 2^16 ----
(a, sorted) = generateArray(2^16)

println("chunk size = 4")
b = @btime cuNativeRankSort(a, 4);
@show checkSorted(b, sorted)

println("chunk size = 16")
b = @btime cuNativeRankSort(a, 16);
@show checkSorted(b, sorted)

println("chunk size = 64")
b = @btime cuNativeRankSort(a, 64);
@show checkSorted(b, sorted)

println("chunk size = 128")
b = @btime cuNativeRankSort(a, 128);
@show checkSorted(b, sorted)

c = @btime cuArrayRankSort(a);
@show checkSorted(c, sorted)

d = @btime sequentialRankSort(a);
@show checkSorted(d, sorted)

# ---- 2^18 ----
(a, sorted) = generateArray(2^18)

println("chunk size = 4")
b = @btime cuNativeRankSort(a, 4);
@show checkSorted(b, sorted)

println("chunk size = 16")
b = @btime cuNativeRankSort(a, 16);
@show checkSorted(b, sorted)

println("chunk size = 64")
b = @btime cuNativeRankSort(a, 64);
@show checkSorted(b, sorted)

println("chunk size = 128")
b = @btime cuNativeRankSort(a, 128);
@show checkSorted(b, sorted)

c = @btime cuArrayRankSort(a);
@show checkSorted(c, sorted)

d = @btime sequentialRankSort(a);
@show checkSorted(d, sorted)


