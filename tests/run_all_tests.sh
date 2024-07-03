#/bin/sh

HOMEDIR=$(pwd)

for scr in $(find . -name "test_*py")
do
    
    cd $(dirname $scr)
    echo "Runing " $scr
    python $(basename $scr)
    if [ "$?" -ne 0 ]
    then
	echo "ERROR!!!"
	exit 1
    fi
    cd $HOMEDIR
    echo
    echo
done