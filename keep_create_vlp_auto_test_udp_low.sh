PROJECT=tpu-prod-env-multipod
ZONE=us-west4-a
NUM_SLICES=1
TPU_TYPE=v5litepod-256
VERSION=v2-alpha-tpuv5-lite
BUCKET_NAME=tonyjohnchen-maxtext

iteration=1

for (( i=0; i<$iteration; i++ ));
do
    RUN_NAME=high_pr_startup_time_${NUM_SLICES}slice_${TPU_TYPE}_$(date +%Y-%m-%d-%H-%M-%S)
    QR_ID=$RUN_NAME
    python3 multihost_job.py --NUM_SLICES=$NUM_SLICES --RUN_NAME=$RUN_NAME --BUCKET_NAME=$BUCKET_NAME --PROJECT=${PROJECT} --ZONE=${ZONE} --CQR_EXTRA_ARGS="--reserved" \
    --TPU_TYPE=$TPU_TYPE --VERSION=$VERSION \
    --COMMAND="bash setup.sh MODE=stable JAX_VERSION=0.4.13 LIBTPU_GCS_PATH=gs://libtpu_internal/tonyjohnchen/viperlite/2023-07-14-17:22:50-libtpu.so; echo \"Sleeping for 60s\" && sleep 60; \
    TPU_LIBRARY_PATH=\$HOME/custom_libtpu/libtpu.so TPU_NAME=local JAX_USE_PJRT_C_API_ON_TPU=1 \
    TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 TPU_VMODULE=tpu_configuration_ops_impl=3 TF_CPP_MIN_LOG_LEVEL=0 \
    EMIT_MEGASCALE_METRICS=True \
    python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
    base_output_directory=gs://max-experiments/ \
    dataset_path=gs://maxtext-dataset/ \
    steps=100 per_device_batch_size=1"
done
gcloud alpha compute tpus queued-resources list --project=${PROJECT} --zone=${ZONE} --filter=startup_time 
