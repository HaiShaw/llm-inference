platform=A100
model=llama2-7b-chat

if [ ! -d  "/data/${model}" ]; then
    mkdir /data/${model}
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-7b-chat-hf/ /data/${model} --recursive
fi

for bs in 1 2 4 8 16; do
    for plen in 32 64  128 256 512 1024 2048; do
        for glen in 16 32 64 128 256; do
            for tp in 1 2 4 8; do
               deepspeed --num_gpus ${tp} ibench_ds.py --platform ${platform} --name /data/${model}  --batch_size ${bs} --prompting_length ${plen} --max_new_tokens ${glen} --performance --ds_inference --use_kernel 2>&1 | tee ./logs/bs_${bs}_plen_${plen}_glen_${glen}_tp_${tp}.log
            done
        done
    done
done

# for bs in 1; do
#     for plen in 32; do
#         for glen in 16; do
#             for tp in 1; do
#                 deepspeed --num_gpus ${tp} ibench_ds.py --platform ${platform} --name /data/${model}  --batch_size ${bs} --prompting_length ${plen} --max_new_tokens ${glen} --performance --ds_inference --use_kernel 2>&1 | tee ./logs/bs_${bs}_plen_${plen}_glen_${glen}_tp_${tp}.log
#             done
#         done
#     done
# done

# python ibench_ds.py --name /data/${model} --platform ${platform} --ds_inference --batch_size 1 --prompting_length 1024 --max_new_tokens 512 --performance
