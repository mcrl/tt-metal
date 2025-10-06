add_library(ttnn_op_experimental_moe ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Moe ALIAS ttnn_op_experimental_moe)

target_precompile_headers(ttnn_op_experimental_moe REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe)

target_sources(
    ttnn_op_experimental_moe
    PRIVATE
        prepare_moe_mapping_tensor/prepare_moe_mapping_tensor.cpp
        prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_op.cpp
        prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_moe PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_moe
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_moe LIBRARY COMPONENT tar)
