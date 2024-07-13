export XLA_FLAGS=--xla_disable_hlo_passes=rematerialization

export tpu=v5e-16

export model_name=llama2-70b
export checkpoint_path=gs://morgandu-tpu/checkpoints/quantized/aqt/llama2-70b-chat
export base_output_dir=gs://morgandu-tpu/mlperf-4.1
export attention=dot_product
export compute_axis_order=0,2,1,3
export prefill_cache_axis_order=0,2,1,3
export ar_cache_axis_order=0,2,1,3

export allow_split_physical_axes=true
export inference_microbenchmark_stages=prefill,generate
export inference_microbenchmark_prefill_lengths=1024

export experiment_time=$(date +%Y-%m-%d-%H-%M)
# export experiment_time=2024-07-09-04-19
echo "export experiment_time=${experiment_time}"

export quant_mode=w-i8-kv-i8
export quantization=int8
export checkpoint_is_quantized=true
export quantize_kvcache=true
export kv_quant_axis=heads_and_dkv
export kv_quant_dtype=int8
export per_device_batch_size=17

export run_name=${model_name}_${tpu}_${attention}_${quant_mode}_${kv_quant_axis}_pbs${per_device_batch_size}_${compute_axis_order//,/}-${prefill_cache_axis_order//,/}-${ar_cache_axis_order//,/}
echo "run_name: ${run_name}"

python3 MaxText/inference_microbenchmark_local_sweep.py \
    MaxText/configs/base.yml \
    tpu=${tpu} \
    quant_mode=${quant_mode} \
    model_name=${model_name} \
    tokenizer_path=assets/tokenizer.llama2 \
    load_parameters_path=${checkpoint_path} \
    async_checkpointing=false \
    weight_dtype=bfloat16 \
    attention=${attention} \
    reshape_q=true \
    scan_layers=false \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    base_output_directory=${base_output_dir}/local_sweep/${tpu}/${model_name}/${quant_mode}/${experiment_time} \
    run_name=${run_name} \
    save_config_to_gcs=true \
    profiler=xplane \
    enable_single_controller=true \
    allow_split_physical_axes=${allow_split_physical_axes} \
    inference_microbenchmark_prefill_lengths=${inference_microbenchmark_prefill_lengths} \
    inference_microbenchmark_stages=${inference_microbenchmark_stages} \
    inference_microbenchmark_loop_iters=10 \
    per_device_batch_size=${per_device_batch_size} \
    quantization=${quantization} \
    quantize_kvcache=${quantize_kvcache} \
    kv_quant_axis=${kv_quant_axis} \
    kv_quant_dtype=${kv_quant_dtype} \
    checkpoint_is_quantized=${checkpoint_is_quantized} \
    compute_axis_order=${compute_axis_order} \
    prefill_cache_axis_order=${prefill_cache_axis_order} \
    ar_cache_axis_order=${ar_cache_axis_order}
