#!/bin/bash

while getopts ":d:r:" option; do
   case $option in
      d) DATAPATH=$OPTARG;;
      r) REPLICATES=$OPTARG;;
      \?) # Invalid option
         echo "Usage: cmd [-d] [-r]"
         exit;;
   esac
done

for i in {1..$REPLICATES}; do
    sbatch -J "ML4SD_G_"${i} -o "ml4sd_G_"${i}".output" -e "ml4sd_G_"${i}".errors" --export=replicate=${i},data_dir=$DATAPATH train_generalist_model.sh
done
