#!/bin/bash

echo 'copying' $WORK/vfeedbacknet-results
rsync -aH $WORK/vfeedbacknet-results/ jemmons@robocop.stanford.edu:/mnt/data/jemmons/vfeedbacknet-results/
