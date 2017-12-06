#include "xtensor/xnoalias.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

// int main(int argc, char const *argv[])
// {
// 	using namespace xt;
// 	using allocator = xsimd::aligned_allocator<double, 32>;
// 	using tensor = xtensor_container<xt::uvector<double, allocator>, 2, layout_type::row_major>;

// 	tensor a = random::rand<double>({3, 3});
// 	tensor b = random::rand<double>({3, 3});
// 	tensor res({ 3, 3 });

// 	for (int i = 0; i < 10000; ++i)
// 	{
// 		xt::noalias(res) = a + b;
// 		// benchmark::DoNotOptimize(res.raw_data());
// 	}
// 	return 0;
// }

int main()
{

	using allocator = xsimd::aligned_allocator<double, 32>;
	using bench_vector = xt::uvector<double, xsimd::aligned_allocator<double, 32>>;
	using batch = xsimd::batch<double, 4>;
	using namespace xt;
	using namespace xsimd;

	bench_vector a(3 * 3);
	bench_vector b(3 * 3);
    std::size_t s = a.size();
    bool is_aligned = false;

    std::size_t sz = 3 * 3;
	bench_vector res(sz);

	for (int j = 0; j < 10000; ++j)
	{
        std::size_t align_begin = is_aligned ? 0 : xsimd::get_alignment_offset(a.data(), s, batch::size);
        std::size_t align_end = align_begin + ((s - align_begin) & ~(batch::size - 1));

		std::size_t i = 0;
		for (; i < align_begin; ++i)
		{
        	res[i] = a[i] + b[i];
		}
        for (; i < align_end; i += batch::size)
        {
            batch blhs(&a[i], aligned_mode());
            batch brhs(&b[i], aligned_mode());
            batch bres = blhs + brhs;
            bres.store_aligned(&res[i]);
        }
		for (; i < s; ++i)
		{
        	res[i] = a[i] + b[i];
		}

        // benchmark::DoNotOptimize(res.data());
	}

	return 0;
}