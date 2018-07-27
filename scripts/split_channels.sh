#!/bin/sh
cd ./data/
mkdir ./signals/dialogues_mono
for i in $(find -name "*mix.wav"); do
	echo "VAR: $i"
	leftChannelEnd="${i%.mix*}.g.wav"
	leftChannel="./signals/dialogues_mono${leftChannelEnd#./signals/dialogues*}"

	rightChannelEnd="${i%.mix*}.f.wav"
	rightChannel="./signals/dialogues_mono${rightChannelEnd#./signals/dialogues*}"
	echo "LEFTCHANNEL: $leftChannel"
	sox $i -b 16 -c 1 $leftChannel remix 1
	sox $i -b 16 -c 1 $rightChannel remix 2

done
