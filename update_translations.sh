#!/bin/bash
PACKAGE="molusce"
SRC_FILES=`find . ! -path "src/$PACKAGE/*" \( -name "*.ui" -o -name "*.py" \) | tr "\n" " "`
TS_FILE=src/$PACKAGE/i18n/*.ts
pylupdate5 -noobsolete $SRC_FILES -ts $TS_FILE