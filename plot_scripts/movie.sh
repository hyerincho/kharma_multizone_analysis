## These are the parameters to control
tag=${1}"/"
ofile=${1}".mp4"
odir="../movies/temporary_frames/"
start_run=0 #116
nruns=1000 #
quantity="log_rho" #"log_beta" #"symlog_u^th" #"symlog_u^phi" #"log_bsq" #"log_Theta" #"log_u" #"log_divB" #"log_sigma" #"symlog_u^r" #"log_bsq" #"Gamma" #
ghostzone=false #
native=false #true # todo: it's weird
overlay_field=false #true #
log_r=true #
FIGURES=false #true

#Note: everything in odir gets deleted first.
rm ${odir}*png

nzones=8 #3 #7 # 2 #

## Add arguments
args=()
if [ $FIGURES == "false" ]; then
  if [[ "$quantity" != *"divB"* ]]; then
    if [[ "$quantity" != *"log_u^"* ]]; then
      if [[ "$quantity" == *"beta"* ]]; then
        # if beta, a different colorbar is needed
        #args+=( '--vmin=-2 --vmax=10' )
        args+=( '--vmin=-4 --vmax=4' )
      elif [[ "$quantity" == *"Gamma"* ]]; then
        args+=( '--vmin=1 --vmax=2' )
      elif [[ "$quantity" == *"Theta"* ]]; then
        args+=( '--vmin=-5 --vmax=2' )
      elif [[ "$quantity" == *"bsq"* ]]; then
        #args+=( '--vmin=-20 --vmax=-6' )
        args+=( '--vmin=-8 --vmax=0' )
      elif [[ "$quantity" == *"sigma"* ]]; then
        args+=( '--vmin=-6 --vmax=1' )
      else
        # unless the quantity is divB or velocity, add colorbar min and max values
        #args+=( '--vmin=-9 --vmax=-7' )
        args+=( '--vmin=-9 --vmax=-1' ) # good for log_rho in n=7
        #args+=( '--vmin=-6 --vmax=-1') # good for log_rho in n=2
        #args+=( '--vmin=-6 --vmax=3' )
        #args+=( '--vmin=-20 --vmax=1' )
      fi
    fi
  fi
  if [[ "$quantity" == *"symlog_u^r"* ]]; then
    args+=( '--vmin=-1e-1 --vmax=1e-1' )
  fi
#if [[ "$quantity" == *"u^th"* ]]; then
#  args+=( '--vmax=1e-12' )
#fi
#if [[ "$quantity" == *"symlog_u^phi"* ]]; then
  #args+=( '--vmin=-1e-7 --vmax=1e-7' ) # good for first outermost annulus
#  args+=( '--vmin=-1e-1 --vmax=1e-1' )
#fi
fi

# don't show black hole
#args+=( '--bh=False' ) 

if [ $ghostzone == true ]; then
  args+=( '-g' )
fi
if [ $native == true ]; then
  args+=( '--native' )
fi
if [ $log_r == true ]; then
  args+=( '--log_r' )
  if [ $FIGURES == false ]; then
    if [ $nzones == 7 ]; then
      args+=( '-sz 7.3' )
    elif [ $nzones == 2 ]; then
      args+=( '-sz 2.8' )
    elif [ $nzones == 3 ]; then
      args+=( '-sz 3.8' )
    elif [ $nzones == 8 ]; then
      args+=( '-sz 8.2' )
    fi
  fi
fi
if [ "$overlay_field" == "true" ] ; then
  args+=( '--overlay_field' )
fi
#args+=( '-sz 16' )

## Run pyharm-movie
iteration=$(($start_run/($nzones-1)+1))
echo "iteration is $iteration"
for (( VAR=${start_run}; VAR<$nruns; VAR++ ))
do
  if [ $nruns -eq 1 ]; then
    dir=""
  else
    dir="bondi_multizone_$(printf %05d ${VAR})"
  fi
  
  # show previous runs in the background
  if [ $log_r == true ]; then
    if [ $VAR -eq 0 ]; then
      fill_num="-1"
    else
      edge=$((($iteration-1)*($nzones-1)))
      fill_num="$edge"
      for i in $(seq $(($edge+1)) $(($VAR-1))); do fill_num="${fill_num},${i}"; done
      if [ $VAR -ge $nzones ]; then
        last_dump_at_this_zone=$((2*($iteration-1)*(${nzones}-1)-${VAR}))
        for i in $(seq $(($edge-($nzones-1))) $(($last_dump_at_this_zone-1))); do fill_num="${fill_num},${i}"; done
      fi
    fi
  else
    fill_num="-1"
  fi

  if [ $VAR -ne 0 ]; then
    if [ $(($VAR % ($nzones-1))) -eq 0 ]; then
      iteration=$(($iteration+1))
    fi
  fi

  echo $VAR: $fill_num
  if [ -d ../data/${tag}/${dir} ]; then
    pyharm-movie $quantity ../data/${tag}/${dir}/*0*.rhdf --output_dir=$odir ${args[@]} --fill=${fill_num} #--tstart=88030402750 # --overlay_field #--tend=327520 -sz 134217728 -g 
  fi
done
python folderToMovie.py ${odir} ../movies/${ofile}
rm ${odir}*png
