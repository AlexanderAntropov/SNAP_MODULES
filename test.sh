#!/bin/bash

lmp_serial -in defect.in -var T 1 | grep "ENERGY and MSD" | awk '{ print $4 " " $5 }' > temp.txt