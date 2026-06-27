#!/bin/bash

gcc main.c ./lib/config.c -fopenmp -I lib -lm

./a.out "./resources/.config"

rm a.out
