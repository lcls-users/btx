#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to launch python scripts from the eLog

  OPTIONS:
    -h|--help
      Definition of options
    -f|--facility
      Facility where we are running at
    -a|--account
      SLURM account (for S3DF)
    -q|--queue
      Queue to use on SLURM
    -n|--ncores
      Number of cores
    -c|--config_file
      Input config file
    -e|--experiment_name
      Experiment Name
    -r|--run_number
      Run Number
    -t|--task
      Task name
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

  case $key in
    -h|--help)
      usage
      exit
      ;;
    -f|--facility)
      FACILITY="$2"
      shift
      shift
      ;;
    -a|--account)
      SLURM_ACCOUNT="$2"
      shift
      shift
      ;;
    -q|--queue)
      QUEUE="$2"
      shift
      shift
      ;;
    -n|--ncores)
      CORES="$2"
      shift
      shift
      ;;
    -c|--config_file)
      CONFIGFILE="$2"
      shift
      shift
      ;;
    -e|--experiment_name)
      EXPERIMENT=$2
      shift
      shift
      ;;
    -r|--run_number)
      RUN_NUM=$2
      shift
      shift
      ;;
    -t|--task)
      TASK="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
    esac
done
set -- "${POSITIONAL[@]}"
if [[ -z ${SLURM_ACCOUNT} ]]; then
    echo "Account not provided, using lcls."
fi

if [[ -z ${FACILITY} ]]; then
    echo "Facility not provided, defaulting to S3DF."
fi

SLURM_ACCOUNT=${SLURM_ACCOUNT:='lcls'}
FACILITY=${FACILITY:='S3DF'}
case $FACILITY in
  'SLAC')
    SIT_PSDM_DATA_DIR='/cds/data/psdm/'
    ANA_CONDA_DIR='/cds/sw/ds/ana/'
    ANA_TOOLS_DIR='/cds/sw/package/'
    SBATCH_CMD_ACCOUNT=''
    ;;
  'SRCF_FFB')
    SIT_PSDM_DATA_DIR='/cds/data/drpsrcf/'
    ANA_CONDA_DIR='/cds/sw/ds/ana/'
    ANA_TOOLS_DIR='/cds/sw/package/'
    SBATCH_CMD_ACCOUNT=''
    ;;
  'S3DF')
    SIT_PSDM_DATA_DIR='/sdf/data/lcls/ds/'
    ANA_CONDA_DIR='/sdf/group/lcls/ds/ana/sw/'
    ANA_TOOLS_DIR='/sdf/group/lcls/ds/tools/'
    SBATCH_CMD_ACCOUNT="#SBATCH -A ${SLURM_ACCOUNT}"
    ;;
  *)
    echo "ERROR! $FACILITY is not recognized."
    ;;
esac

if [[ -z ${TASK} || -z ${CONFIGFILE} ]]; then
    echo "You must provide a config yaml and task choice."
    usage
    exit
fi

QUEUE=${QUEUE:='milano'}
CORES=${CORES:=1}
# TODO: find_peaks needs to be handled from ischeduler. For now we do this...
if [ ${TASK} != 'find_peaks' ] &&\
   [ ${TASK} != 'find_peaks_by_peaknet' ] &&\
   [ ${TASK} != 'stream_analysis' ] &&\
   [ ${TASK} != 'determine_cell' ] &&\
   [ ${TASK} != 'opt_geom' ]; then
  CORES=1
fi

EXPERIMENT=${EXPERIMENT:='None'}
RUN_NUM=${RUN_NUM:='None'}
THIS_CONFIGFILE=${CONFIGFILE}
if [ ${RUN_NUM} != 'None' ]; then
  THIS_CONFIGFILE="${CONFIGFILE%.*}_${RUN_NUM}.yaml"
fi

ANA_CONDA_MANAGE="${ANA_CONDA_DIR}conda1/manage/bin/"
## ANA_CONDA_BIN="${ANA_CONDA_DIR}conda1/inst/envs/ana-4.0.47-py3/bin/"
ANA_CONDA_BIN="/cds/sw/package/conda_envs/cwang31/peaknet-1.0/bin/"
WHICHPYTHON="${ANA_CONDA_BIN}python"
WHICHMPIRUN="${ANA_CONDA_BIN}mpirun"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_PY="${SCRIPT_DIR}/main.py"
if [ ${CORES} -gt 1 ]; then
MAIN_PY="${WHICHMPIRUN} ${MAIN_PY}"
else
MAIN_PY="${WHICHPYTHON} ${MAIN_PY}"
fi

UUID=$(cat /proc/sys/kernel/random/uuid)
if [ "${HOME}" == '' ]; then
  TMP_DIR="${SCRIPT_DIR}"
else
  TMP_DIR="${HOME}/.btx/"
fi
mkdir -p $TMP_DIR
TMP_EXE="${TMP_DIR}/task_${UUID}.sh"

#Submit to SLURM
sbatch << EOF
#!/bin/bash

${SBATCH_CMD_ACCOUNT}
#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name ${TASK}
#SBATCH --ntasks=${CORES}
#SBATCH --gres=gpu:1080ti:1

source "${ANA_CONDA_MANAGE}psconda.sh"
conda activate /cds/sw/package/conda_envs/cwang31/peaknet-1.0
conda env list | grep '*'
which mpirun
which python
export FACILITY=${FACILITY}
export SIT_PSDM_DATA=${SIT_PSDM_DATA_DIR}
export PATH="${ANA_TOOLS_DIR}/crystfel/0.10.2/bin:$PATH"
export PYTHONPATH="${PYTHONPATH}:$( dirname -- ${SCRIPT_DIR})"
export NCORES=${CORES}
export TMP_EXE=${TMP_EXE}
export WHICHPYTHON="${WHICHPYTHON}"

if [ ${RUN_NUM} != 'None' ]; then
  echo "new config file: ${THIS_CONFIGFILE}"
  sed "s/run:/run: ${RUN_NUM} #/g" ${CONFIGFILE} > ${THIS_CONFIGFILE}
fi

echo "$MAIN_PY -c ${THIS_CONFIGFILE} -t $TASK"
$MAIN_PY -c ${THIS_CONFIGFILE} -t $TASK
if [ ${RUN_NUM} != 'None' ]; then
  rm -f ${THIS_CONFIGFILE}
fi
EOF

echo "Job sent to queue"
