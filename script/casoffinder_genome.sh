#!/bin/sh

target=$1
genome=$2
MM=$3
device=$4
tm=$5
touch .temp_${tm}_casoffinder.in
echo "${genome}" >> .temp_${tm}_casoffinder.in
echo "NNNNNNNNNNNNNNNNNNNNNGG" >> .temp_${tm}_casoffinder.in
echo "${target}NNN ${MM}" >> .temp_${tm}_casoffinder.in

cas-offinder .temp_${tm}_casoffinder.in ${device} .temp_${tm}_casoffinder.out
rm .temp_${tm}_casoffinder.in
