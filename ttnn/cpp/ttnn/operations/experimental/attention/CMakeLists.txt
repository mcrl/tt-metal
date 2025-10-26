add_library(ttnn_op_experimental_attention ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Attention ALIAS ttnn_op_experimental_attention)

target_precompile_headers(ttnn_op_experimental_attention REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_attention)

target_sources(
    ttnn_op_experimental_attention
    PRIVATE
        extract_attention_input_prefill/extract_attention_input_prefill.cpp
        extract_attention_input_prefill/device/extract_attention_input_prefill_op.cpp
        extract_attention_input_prefill/device/extract_attention_input_prefill_program_factory.cpp
        extract_attention_input_decode/extract_attention_input_decode.cpp
        extract_attention_input_decode/device/extract_attention_input_decode_op.cpp
        extract_attention_input_decode/device/extract_attention_input_decode_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_attention PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_attention
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_attention LIBRARY COMPONENT tar)
