#!/bin/bash
# Sequential run script, logs stdout (and stderr) for subsequent parsing

run () {
    echo "$1"
    eval "$1 > ${1%.sh}.stdout 2> ${1%.sh}.stderr"
}

while read sh; do
    run $sh
done <$1

