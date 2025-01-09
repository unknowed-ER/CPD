source activate lla
ocp_memory=2000
var1=1
var2=0
while (( $var1 ))
do
    sleep 1 # Waits 0.1 second.
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt $ocp_memory ]; then
            if (( $var2 )); then
                echo 'run... '
                source $1
                var1=0
                var2=0
                break
            else
                echo 'checking gpu... '
                var2=1
                sleep 10
            fi
        else
            var2=0
            echo 'waiting for available gpu...'
        fi
        count=$(($count+1))    
    done    
done