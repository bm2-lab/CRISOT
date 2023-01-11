#!/bin/sh

target=$1
genome=$2
MM=$3
device=$4
touch .temp_casoffinder.in
echo "${genome}" >> .temp_casoffinder.in
echo "NNNNNNNNNNNNNNNNNNNNNGG" >> .temp_casoffinder.in
echo "${target}NNN ${MM}" >> .temp_casoffinder.in

cas-offinder .temp_casoffinder.in ${device} .temp_casoffinder.out
rm .temp_casoffinder.in
