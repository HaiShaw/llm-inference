pip install deepspeed accelerate sentencepiece transformers

if [ ! -d  "/data/llama2-70b-chat" ]; then
    mkdir /data/llama2-70b-chat
    pip3 --no-cache-dir install --upgrade awscli
    export AWS_ACCESS_KEY_ID=id_string
    export AWS_SECRET_ACCESS_KEY=key_string
    aws --region=us-east-2 s3 cp s3://datasets.dl/llama-2/llama-2-70b-chat-hf/ /data/llama2-70b-chat --recursive
fi

platform=MI300X

python ibench_hf.py --model llama2-70b-chat --platform ${platform} --n 1
