METHODS="point-cloud nearest linear clough-tocher gaussian"
NUM_POINTS_SET="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144"
echo "SOLVE.SH: Generating True Fields"
ipython generate-true-fields.ipynb
for NUM_POINTS in $( echo $NUM_POINTS_SET ); do
    for METHOD in $( echo $METHODS ); do
        echo "SOLVE.SH: Using $NUM_POINTS points with $METHOD method"
        ipython solve.ipynb $NUM_POINTS $METHOD
    done
done
ipython analyse.ipynb