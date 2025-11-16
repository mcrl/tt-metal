#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::paged_cache::detail {

tt::tt_metal::operation::ProgramWithCallbacks batched_paged_fill_cache_multi_core(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table_tensor,
    uint32_t batch_size);

}