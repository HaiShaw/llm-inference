platform=H100
model=llama2-70b-chat

if [ ! -d  "/data/${model}" ]; then
    mkdir /data/${model}
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-7b-chat-hf/ /data/${model} --recursive
fi

rpd_file="${platform}_${model}.rpd"
json_file="${platform}_${model}.json"

echo ${rpd_file}
echo ${json_file}

if [ -f "$rpd_file" ] ; then
    rm "$rpd_file"
fi
python -m rocpd.schema --create $rpd_file

python ibench_hf.py --model ${model} --platform ${platform}

if [ -f "$json_file" ] ; then
    rm "$json_file"
fi
python /workspace/rocmProfileData/tools/rpd2tracing.py $rpd_file $json_file
