add_library(ttnn_op_experimental_moe ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Moe ALIAS ttnn_op_experimental_moe)

target_precompile_headers(ttnn_op_experimental_moe REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe)

target_sources(
    ttnn_op_experimental_moe
    PRIVATE
        prepare_moe_routing_tensors/prepare_moe_routing_tensors.cpp
        prepare_moe_routing_tensors/device/prepare_moe_routing_tensors_op.cpp
        prepare_moe_routing_tensors/device/prepare_moe_routing_tensors_program_factory.cpp
        projection_to_intermediate/projection_to_intermediate.cpp
        projection_to_intermediate/device/projection_to_intermediate_op.cpp
        projection_to_intermediate/device/projection_to_intermediate_program_factory.cpp
        projection_to_output/projection_to_output.cpp
        projection_to_output/device/projection_to_output_op.cpp
        projection_to_output/device/projection_to_output_program_factory.cpp
        scatter_moe_input/scatter_moe_input.cpp
        scatter_moe_input/device/scatter_moe_input_op.cpp
        scatter_moe_input/device/scatter_moe_input_program_factory.cpp
        local_reduce_moe_output/local_reduce_moe_output.cpp
        local_reduce_moe_output/device/local_reduce_moe_output_op.cpp
        local_reduce_moe_output/device/local_reduce_moe_output_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_moe PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_moe
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_moe LIBRARY COMPONENT tar)
