using CUDA
using BenchmarkTools

CUDA.versioninfo()
@show CUDA.functional()

# Generate an array in random order and return the array and its sorted version
# Using for testing purposes
function generateArray(n::Int64)
  a = rand(n)
  return a, sort(a)
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
  sorted = similar(a)
  temp_gpu::CuArray{Int64, 1} = CuArray(zeros(UInt8, n))

  for i in 1:n
    CUDA.@allowscalar temp_gpu[1:n] .= a_gpu[i] .> a_gpu[1:n]
    # sum of 1s in temp_gpu
    idx = sum(Array(temp_gpu))
    sorted[idx+1] = a[i]
  end
  return sorted
end

(a, sorted) = generateArray(2^16)

c = @btime cuArrayRankSort(a)
d = @btime sequentialRankSort(a)

# Check for correctness
@show checkSorted(c, sorted)
@show checkSorted(d, sorted)
