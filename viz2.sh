#!/bin/bash

# Define the subject variable
SUBJECT="S11"
OUTPUT_DIR="outputs_${SUBJECT}"

# Create the directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Define the list of actions
ACTIONS=('Directions 1' 'Discussion 1' 'Discussion 2' 'Eating 1' 'Eating' 
'Greeting 2' 'Greeting' 'Phoning 2' 'Phoning 3' 'Photo 1' 'Photo' 
'Posing 1' 'Posing' 'Purchases 1' 'Purchases' 'Sitting 1' 'Sitting' 
'SittingDown 1' 'SittingDown' 'Smoking 2' 'Smoking' 'Waiting 1' 'Waiting' 
'WalkDog 1' 'WalkDog' 'Walking 1' 'Walking' 'WalkTogether 1' 'WalkTogether')

# Loop over each action and run the command
for ACTION in "${ACTIONS[@]}"; do
    # Replace spaces with underscores for the output filename
    FILENAME_ACTION=$(echo ${ACTION} | tr ' ' '_')
    OUTPUT_FILE="${OUTPUT_DIR}/${SUBJECT}_${FILENAME_ACTION}.gif"
    
    # Run the python script with the current action
    python viz.py --architecture gcn --evaluate checkpoint/2023-08-22T02:21:50.476018/ckpt_best.pth.tar --viz_subject ${SUBJECT} --viz_action "${ACTION}" --viz_camera 0 --viz_output ${OUTPUT_FILE} --viz_size 3 --viz_downsample 2 --viz_limit 160
done
