platform=MI300X
model=llama2-7b-chat

if [ ! -d  "/data/${model}" ]; then
    mkdir /data/${model}
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-7b-chat-hf/ /data/${model} --recursive
fi

deepspeed --num_gpus 2 ibench_ds.py --platform ${platform} --name /data/${model}  --batch_size 1 --prompting_length 32 --performance --ds_inference --max_new_tokens 64 --use_kernel
# python ibench_ds.py --name /data/${model} --platform ${platform} --ds_inference --batch_size 1 --prompting_length 1024 --max_new_tokens 512 --performance