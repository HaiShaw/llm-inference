platform=MI300X
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