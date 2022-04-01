#!/bin/sh

FONT=Palatino

DS11=1.0 # display start
DS12=4.0 # display start
DS21=8.0
DS22=11.0
DS3=13.0

DE=17.0 # display end

FID1=2 # fade in duration

DE1=6
DE2=16
DE3=16

FID2=1
FID3=0

FOD=1 # fade out duration

DINA=15

TANIM=200
FEND=5
TSTOP=$(($TANIM + $DINA + 2))
TEND=$(($TSTOP + $FEND + 3))

FOCUS=$1

LINE1="COVID-19\\\: Mar 2020 - Mar 2022"
LINE21="This is the course of the pandemic in $FOCUS"
LINE22="out of 225 countries recorded."
LINE3='Data source\\\: OurWorldInData.org'

echo "Producing video $TEND seconds long; animation stops at $TSTOP"

ffmpeg \
    -f lavfi -i color=black:1400x1200 \
    -f lavfi -r 24 \
    -i color=black:1400x1200 \
    -i animation-${FOCUS}.mp4 \
    -filter_complex "
    [1]split=2[mv0][t11];
    chromakey=black,
    split=4[t12][t21][t22][t3];

    [t11]drawtext=font=$FONT:fontcolor=FFFFFF:fontsize=64:text=$LINE1: x='(main_w-text_w)/2': y='(main_h-text_h)/3',
    fade=t=in:st=$DS11:d=$FID1, fade=t=out:st=$DE1:d=$FOD [s11];
    [t12]drawtext=font=$FONT:fontcolor=FFFFFF:fontsize=64:text=$FOCUS: x='(main_w-text_w)/2': y='(main_h-text_h)/2':enable='between(t,$DS12,$DE)',
    fade=t=out:st=$DE1:d=$FOD  [s12];
    [t21]drawtext=font=$FONT:fontcolor=FFFFFF:fontsize=42:text=$LINE21: x='(main_w-text_w)/2': y='(main_h*2/5-text_h/2)':enable='between(t,$DS21,$DE)',
    fade=t=in:st=$DS21:d=$FID2, fade=t=out:st=$DE2:d=$FOD  [s21] ;
    [t22]drawtext=font=$FONT:fontcolor=FFFFFF:fontsize=42:text=$LINE22: x='(main_w-text_w)/2': y='(main_h*2.5/5-text_h/2)':enable='between(t,$DS21,$DE)',
    fade=t=in:st=$DS22:d=$FID2, fade=t=out:st=$DE2:d=$FOD  [s22];
    [t3]drawtext=font=$FONT:fontcolor=FFFFFF:fontsize=18:text=$LINE3: x='(main_w-text_w)/2': y='(main_h-text_h)*4.8/5',
    fade=t=in:st=$DS3:d=$FID3, fade=t=out:st=$DE3:d=$FOD  [s3];

    [mv0][s11]overlay=x='0':y='0'[mv11];
    [mv11][s12]overlay=x='0':y='0'[mv12];
    [mv12][s21]overlay=x='0':y='0'[mv21];
    [mv21][s22]overlay=x='0':y='0'[mv22];
    [mv22][s3]overlay=0:0[intro];

    [intro] format=pix_fmts=yuva420p,fade=t=out:st=$DINA:d=3:alpha=1,setpts=PTS-STARTPTS[va0];
    [2:v] format=pix_fmts=yuva420p,fade=t=in:st=0:d=7:alpha=1,setpts=PTS-STARTPTS+$DINA/TB,fifo[va1];

    [0:v][va0]overlay[over1];
    [over1][va1]overlay=format=yuv420[out];
    [out]fade=t=out:st=${TSTOP}:d=$FEND [fin]" \
-c:v libx264 -c:a copy -t $TEND -map [fin] ./examples/covid-${FOCUS}-barchart.mp4
