#!/bin/bash

PATH="/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
BASENAME="${0##*/}"

usage () {
  if [ "${#@}" -ne 0 ]; then
    echo "* ${*}"
    echo
  fi
  cat <<ENDUSAGE
Usage:

export BATCH_FILE_TYPE="script"
export BATCH_FILE_S3_URL="s3://my-bucket/my-script"
${BASENAME} script-from-s3 [ <script arguments> ]

  - or -

export BATCH_FILE_TYPE="zip"
export BATCH_FILE_S3_URL="s3://my-bucket/my-zip"
${BASENAME} script-from-zip [ <script arguments> ]
ENDUSAGE

  exit 2
}

# Standard function to print an error and exit with a failing return code
error_exit () {
  echo "${BASENAME} - ${1}" >&2
  exit 1
}

# Check what environment variables are set
if [ -z "${BATCH_FILE_TYPE}" ]; then
  usage "BATCH_FILE_TYPE not set, unable to determine type (zip/script) of URL ${BATCH_FILE_S3_URL}"
fi

if [ -z "${BATCH_FILE_S3_URL}" ]; then
  usage "BATCH_FILE_S3_URL not set. No object to download."
fi

scheme="$(echo "${BATCH_FILE_S3_URL}" | cut -d: -f1)"
if [ "${scheme}" != "s3" ]; then
  usage "BATCH_FILE_S3_URL must be for an S3 object; expecting URL starting with s3://"
fi

# Check that necessary programs are available
which aws >/dev/null 2>&1 || error_exit "Unable to find AWS CLI executable."
which unzip >/dev/null 2>&1 || error_exit "Unable to find unzip executable."

# Create a temporary directory to hold the downloaded contents, and make sure
# it's removed later, unless the user set KEEP_BATCH_FILE_CONTENTS.
cleanup () {
   if [ -z "${KEEP_BATCH_FILE_CONTENTS}" ] \
     && [ -n "${TMPDIR}" ] \
     && [ "${TMPDIR}" != "/" ]; then
      rm -r "${TMPDIR}"
   fi
}
trap 'cleanup' EXIT HUP INT QUIT TERM
# mktemp arguments are not very portable.  We make a temporary directory with
# portable arguments, then use a consistent filename within.
#TMPDIR="$(mktemp -d -t tmp.XXXXXXXXX)" || error_exit "Failed to create temp directory."
TMPDIR="/root" 
TMPFILE="${TMPDIR}/batch-file-temp"
install -m 0600 /dev/null "${TMPFILE}" || error_exit "Failed to create temp file."

# Fetch and run a script
fetch_and_run_script () {
  # Create a temporary file and download the script
  aws s3 cp "${BATCH_FILE_S3_URL}" - > "${TMPFILE}" || error_exit "Failed to download S3 script."

  # Make the temporary file executable and run it with any given arguments
  local script="./${1}"; shift
  chmod u+x "${TMPFILE}" || error_exit "Failed to chmod script."
  exec ${TMPFILE} "${@}" || error_exit "Failed to execute script."
}

# Download a zip and run a specified script from inside
fetch_and_run_zip () {
  # Create a temporary file and download the zip file
  aws s3 cp "${BATCH_FILE_S3_URL}" - > "${TMPFILE}" || error_exit "Failed to download S3 zip file from ${BATCH_FILE_S3_URL}"

  # Create a temporary directory and unpack the zip file
  cd "${TMPDIR}" || error_exit "Unable to cd to temporary directory."
  unzip -q "${TMPFILE}" || error_exit "Failed to unpack zip file."

  # Use first argument as script name and pass the rest to the script
  local script="./${1}"; shift
  [ -r "${script}" ] || error_exit "Did not find specified script '${script}' in zip from ${BATCH_FILE_S3_URL}"
  chmod u+x "${script}" || error_exit "Failed to chmod script."
  exec "${script}" "${@}" || error_exit " Failed to execute script."
}

# Download model weights
download_model_weights () {
  # Create a temporary file and download the zip file
  aws s3 cp s3://seelig.gun/batch_data/model_weights/evodiff_msa/msa-oaar-maxsub.tar /root/.cache/torch/hub/checkpoints/msa-oaar-maxsub.tar || error_exit "Failed to download EvoDiff-MSA model weights"
  aws s3 cp s3://seelig.gun/batch_data/model_weights/esmmsa/esm_msa1b_t12_100M_UR50S.pt /root/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S.pt || error_exit "Failed to download MSA Transformer model weights"
  aws s3 cp s3://seelig.gun/batch_data/model_weights/esmmsa/esm_msa1b_t12_100M_UR50S-contact-regression.pt /root/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S-contact-regression.pt || error_exit "Failed to download MSA Transformer model weights 2"
  mkdir -p /root/.cache/torch/hub/checkpoints
  aws s3 sync s3://seelig.gun/batch_data/model_weights/proteinmpnn /root/proteinmpnn_weights || error_exit "Failed to download ProteinMPNN model weights"
}

download_model_weights_eu () {
  # Create a temporary file and download the zip file
  aws s3 cp s3://seelig.gun.eu/batch_data/model_weights/evodiff_msa/msa-oaar-maxsub.tar /root/.cache/torch/hub/checkpoints/msa-oaar-maxsub.tar  --region eu-south-2 || error_exit "Failed to download EvoDiff-MSA model weights"
  aws s3 cp s3://seelig.gun.eu/batch_data/model_weights/esmmsa/esm_msa1b_t12_100M_UR50S.pt /root/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S.pt --region eu-south-2 || error_exit "Failed to download MSA Transformer model weights"
  aws s3 cp s3://seelig.gun.eu/batch_data/model_weights/esmmsa/esm_msa1b_t12_100M_UR50S-contact-regression.pt /root/.cache/torch/hub/checkpoints/esm_msa1b_t12_100M_UR50S-contact-regression.pt --region eu-south-2 || error_exit "Failed to download MSA Transformer model weights 2"
  mkdir -p /root/.cache/torch/hub/checkpoints
  aws s3 sync s3://seelig.gun.eu/batch_data/model_weights/proteinmpnn /root/proteinmpnn_weights --region eu-south-2 || error_exit "Failed to download ProteinMPNN model weights"
}

if [ "$REGION" == "US" ]; then
    download_model_weights "${@}"
elif [ "$REGION" == "EU" ]; then
    download_model_weights_eu "${@}"
fi

# Main - dispatch user request to appropriate function
case ${BATCH_FILE_TYPE} in
  zip)
    if [ ${#@} -eq 0 ]; then
      usage "zip format requires at least one argument - the script to run from inside"
    fi
    fetch_and_run_zip "${@}"
    ;;

  script)
    fetch_and_run_script "${@}"
    ;;

  *)
    usage "Unsupported value for BATCH_FILE_TYPE. Expected (zip/script)."
    ;;
esac
