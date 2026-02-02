#!/usr/bin/env bash
# this is a container runner for using the ros master on the robot. 
# If you are just testing basic functions on the host machine, you just need to use run.sh
# Launch the workspace container while pointing ROS to an external robot master.
# Usage: run_with_robot.sh --master HOST[:PORT] [--advertise HOST_OR_IP]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/run.sh"

say() { printf "\033[1;36m[run_with_robot]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[run_with_robot]\033[0m %s\n" "$*"; }
usage() {
  cat <<'USAGE'
Usage: ./run_with_robot.sh --master HOST[:PORT] [--advertise HOST_OR_IP]

Options:
  --master, -m       Robot ROS master host (optionally host:port). Required.
  --master-uri       Full ROS master URI (e.g. http://robot:11311).
                     Overrides --master if provided.
  --advertise, -a    Hostname or IP that other nodes (robot) should use to reach
                     this container. Defaults to the first non-loopback IP.
  --help, -h         Show this message.

Environment overrides:
  ROBOT_MASTER, ROBOT_MASTER_URI, ROS_ADVERTISE may be set instead of flags.

The script exports ROS_MASTER_URI, ROS_HOSTNAME, and ROS_IP before deferring to
run.sh, which handles GPU detection and launches the container.
USAGE
}

ROBOT_MASTER="${ROBOT_MASTER:-}"
ROBOT_MASTER_URI="${ROBOT_MASTER_URI:-}"
ROS_ADVERTISE="${ROS_ADVERTISE:-}"  # optional env variable

while (( "$#" )); do
  case "$1" in
    --master|-m)
      [[ $# -ge 2 ]] || { echo "--master requires an argument" >&2; usage >&2; exit 1; }
      ROBOT_MASTER="$2"
      shift 2
      ;;
    --master-uri)
      [[ $# -ge 2 ]] || { echo "--master-uri requires an argument" >&2; usage >&2; exit 1; }
      ROBOT_MASTER_URI="$2"
      shift 2
      ;;
    --advertise|-a)
      [[ $# -ge 2 ]] || { echo "--advertise requires an argument" >&2; usage >&2; exit 1; }
      ROS_ADVERTISE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ROBOT_MASTER_URI" ]]; then
  if [[ -z "$ROBOT_MASTER" ]]; then
    echo "Error: --master or --master-uri must be provided (or ROBOT_MASTER[_URI])." >&2
    usage >&2
    exit 1
  fi
  master_host="${ROBOT_MASTER%%:*}"
  master_port="${ROBOT_MASTER##*:}"
  if [[ "$master_host" == "$master_port" ]]; then
    master_port="11311"
  fi
  if [[ -z "$master_host" ]]; then
    echo "Error: could not parse master host from '$ROBOT_MASTER'." >&2
    exit 1
  fi
  ROBOT_MASTER_URI="http://${master_host}:${master_port}"
fi

if [[ -z "$ROS_ADVERTISE" ]]; then
  auto_ip="$(hostname -I 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i!="127.0.0.1") {print $i; exit}}')"
  if [[ -n "$auto_ip" ]]; then
    ROS_ADVERTISE="$auto_ip"
    warn "Advertise address not supplied; using autodetected ${ROS_ADVERTISE}."
  else
    echo "Error: unable to determine a non-loopback IP; please provide --advertise." >&2
    exit 1
  fi
fi

say "ROS_MASTER_URI = ${ROBOT_MASTER_URI}"
say "Advertise address = ${ROS_ADVERTISE}"

export ROS_MASTER_URI="$ROBOT_MASTER_URI"
export ROS_HOSTNAME="$ROS_ADVERTISE"
export ROS_IP="$ROS_ADVERTISE"

if [[ ! -x "$RUN_SH" ]]; then
  echo "Error: cannot execute ${RUN_SH}" >&2
  exit 1
fi

exec "$RUN_SH"
