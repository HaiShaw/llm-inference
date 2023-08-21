platform=MI300X
model=llama2-7b-chat

if [ ! -d  "/data/${model}" ]; then
    mkdir /data/${model}
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-7b-chat-hf/ /data/${model} --recursive
fi

rpd_file="${platform}_${model}_ds.rpd"
json_file="${platform}_${model}_ds.json"

echo ${rpd_file}
echo ${json_file}

if [ -f "$rpd_file" ] ; then
    rm "$rpd_file"
fi
python -m rocpd.schema --create $rpd_file

deepspeed --num_gpus 8 ibench_ds.py --platform ${platform} --name /data/${model}  --batch_size 1 --prompting_length 512 --performance --ds_inference --max_new_tokens 64 --use_kernel
# python ibench_ds.py --name /data/${model} --platform ${platform} --ds_inference --batch_size 1 --prompting_length 1024 --max_new_tokens 512 --performance

if [ -f "$json_file" ] ; then
    rm "$json_file"
fi
python /workspace/rocmProfileData/tools/rpd2tracing.py $rpd_file $json_file
