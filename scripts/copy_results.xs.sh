#!/bin/bash

echo 'copying' $WORK/vfeedbacknet-results
rsync -avH $WORK/vfeedbacknet-results/ jemmons@robocop.stanford.edu:/mnt/data/jemmons/vfeedbacknet-results/
