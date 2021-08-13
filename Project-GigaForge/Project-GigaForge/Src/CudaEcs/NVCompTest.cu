#include "nvcomp/cascaded.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include <cassert>
#include <cstdlib>
#include <vector>
#include "CudaGlobals.h"
#include "../../Timer.h"
#include "../CudaCommon/Bitset.h"
#define CUDA_CHECK checkCudaError
#define REQUIRE(x) if(!(x)){throw "Error";}else{}
using namespace std;
using namespace nvcomp;
#define Lz4
#undef Lz4
void PrintMb(std::string message, float size)
{
	std::cout << message << ": " << size / 1024.0f / 1024 << " MB" << std::endl;
}

template <typename T>
void test_lz4(T* input, int uncompressedSize, const size_t chunk_size = 1 << 16)
{
	cudaFree(0);//Takes 250ms to initialize
	auto uncompressedBytes = uncompressedSize * sizeof(T);
	PrintMb("Data uncompressedSize", uncompressedBytes);
	Timer t = Timer();
	// create GPU only input buffer
	T* d_in_data;
	const size_t in_bytes = sizeof(T) * uncompressedSize;
	CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
	t.Restart("Allocate on gpu");
	CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));
	t.Restart("Copy to device");
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	size_t comp_temp_bytes = 0;
	size_t comp_out_bytes = 0;
	void* d_comp_temp;
	void* d_comp_out;

#ifdef Lz4
	LZ4Compressor compressor(chunk_size);
#else
	CascadedCompressor compressor(TypeOf<T>(), 1, 0, false);
#endif

	compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
	REQUIRE(comp_temp_bytes > 0);
	REQUIRE(comp_out_bytes > 0);

	// allocate temp buffer
	CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

	// Allocate output buffer
	CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

	size_t* comp_out_bytes_ptr;
	cudaMalloc((void**)&comp_out_bytes_ptr, sizeof(size_t));
	CUDA_CHECK(cudaDeviceSynchronize());
	t.Restart("Intermediate");
	compressor.compress_async(d_in_data, in_bytes, d_comp_temp, comp_temp_bytes, d_comp_out, comp_out_bytes_ptr, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	auto compressTime = t.Restart("Compress");
	CUDA_CHECK(cudaMemcpy(
		&comp_out_bytes,
		comp_out_bytes_ptr,
		sizeof(comp_out_bytes),
		cudaMemcpyDeviceToHost));
	cudaFree(comp_out_bytes_ptr);

	PrintMb("Out data uncompressedSize", comp_out_bytes);
	std::cout << "Ratio: " << (float)uncompressedBytes / comp_out_bytes << std::endl;
	PrintMb("CompressThoughput", (float)uncompressedBytes / (compressTime / 1000.0f));
	cudaFree(d_comp_temp);
	cudaFree(d_in_data);
	// Test to make sure copying the compressed file is ok
	void* copied = 0;
	CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
	CUDA_CHECK(cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
	cudaFree(d_comp_out);
	d_comp_out = copied;

#ifdef Lz4
	LZ4Decompressor decompressor;
#else
	CascadedDecompressor decompressor;
#endif

	size_t decomp_temp_bytes;
	size_t decomp_out_bytes;
	decompressor.configure(
		d_comp_out,
		comp_out_bytes,
		&decomp_temp_bytes,
		&decomp_out_bytes,
		stream);

	void* d_decomp_temp;
	cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

	T* out_ptr;
	cudaMalloc(&out_ptr, decomp_out_bytes);

	// make sure the data won't match input if not written to, so we can verify
	// correctness
	cudaMemset(out_ptr, 0, decomp_out_bytes);
	CUDA_CHECK(cudaStreamSynchronize(stream))
	t.Restart("Before decompress");
	decompressor.decompress_async(
		d_comp_out,
		comp_out_bytes,
		d_decomp_temp,
		decomp_temp_bytes,
		out_ptr,
		decomp_out_bytes,
		stream);
	CUDA_CHECK(cudaStreamSynchronize(stream))
	auto decompressTime = t.Restart("Decompress");
	PrintMb("DecompressThoughput", (float)uncompressedBytes / (decompressTime / 1000.0f));


	// Copy result back to host
	T* res = new T[uncompressedSize];
	cudaMemcpy(res, out_ptr, uncompressedSize * sizeof(T), cudaMemcpyDeviceToHost);
	t.Restart("Copy to host");

	// Verify correctness
	// REQUIRE(res == input);
	for (size_t i = 0; i < uncompressedSize; i++)
	{
		REQUIRE(res[i] == input[i]);
	}

	cudaFree(d_comp_out);
	cudaFree(out_ptr);
	cudaFree(d_decomp_temp);
	delete res;
}

void TestComp()
{
	auto t = Timer();
	using T = bool;
	int size = 1 << 30;
	auto bitset = GigaEntity::Bitset(size);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		auto bo = sin(i / 100) > 0;
		auto rand = std::rand();
		if (rand < RAND_MAX / 50) bo = true;
		bitset.unpackedData[i] = bo;
	}
	bitset.Pack();
	test_lz4(bitset.packedData, bitset.packedSize);
	// test_lz4(bitset.unpackedData, bitset.unpackedSize);
	t.Stop("total");
}
