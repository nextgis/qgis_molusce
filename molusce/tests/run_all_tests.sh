#/bin/sh

HOMEDIR=$(pwd)

for scr in $(find . -name "test_*py")
do
    
    cd $(dirname $scr)
    echo "Runing" $scr
    python $(basename $scr)
    if [ "$?" -eq 0 ]
    then
	echo ok
    else
	echo "ERROR!!!"
	exit 1
    fi
    cd $HOMEDIR
    echo
    echo
done