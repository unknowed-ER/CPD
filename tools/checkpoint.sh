subdir=`ls -d ./*/`
for dir in $subdir
do
   echo $dir
   cp adapter_config.json $dir
   mv $dir/pytorch_model.bin $dir/adapter_model.bin
done