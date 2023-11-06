#!/usr/bin/bash

rounds_per_run=10
total_rounds=50
iterations=$total_rounds/$rounds_per_run
for (( i=1; i <= iterations; ++i ))
do
  python -m unlearning_fl.main --multirun num_rounds=$rounds_per_run sanitized_dataset=True,False
done

rounds_per_run=25
total_rounds=25
iterations=$total_rounds/$rounds_per_run

for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
done