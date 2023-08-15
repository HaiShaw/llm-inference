platform=MI300X

if [ ! -d  "/data/llama2-70b-chat" ]; then
    mkdir /data/llama2-70b-chat
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=id_string
    export AWS_SECRET_ACCESS_KEY=key_string
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-70b-chat-hf/ /data/llama2-70b-chat --recursive
fi

rpd_file="${platform}_llama2.rpd"
json_file="${platform}_llama2.json"

echo ${rpd_file}
echo ${json_file}

if [ -f "$rpd_file" ] ; then
    rm "$rpd_file"
fi
python -m rocpd.schema --create $rpd_file

python ibench_hf.py --model llama2-70b-chat --platform ${platform} --n 1

if [ -f "$json_file" ] ; then
    rm "$json_file"
fi
python /workspace/rocmProfileData/tools/rpd2tracing.py $rpd_file $json_file