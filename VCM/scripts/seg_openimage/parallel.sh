#!/bin/bash

MAX_PARALLEL=2

run () {
    echo "$1"
    sem -j $MAX_PARALLEL bash "$1 > ${1%.sh}.stdout 2> ${1%.sh}.stderr"
}

while read sh; do
    run $sh
done <$1
sem --wait

