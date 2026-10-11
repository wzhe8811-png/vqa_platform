#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/root/miniconda3/envs/llava_env/bin/python"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
BASE_DIR="/root/autodl-tmp/vision_qa_platform/agent_workspace"
RUN_DIR="${BASE_DIR}/.run"
LOG_DIR="${BASE_DIR}/.logs"

TRACE_NAME="trace_webapp"
TRACE_DIR="${BASE_DIR}/trace_webapp"
TRACE_PORT="7860"
TRACE_HEALTH_PATH="/"
TRACE_PID_FILE="${RUN_DIR}/${TRACE_NAME}.pid"
TRACE_LOG_FILE="${LOG_DIR}/${TRACE_NAME}.log"

CHAT_NAME="chat_vqa_webapp"
CHAT_DIR="${BASE_DIR}/chat_vqa_webapp"
CHAT_PORT="7861"
CHAT_HEALTH_PATH="/api/health"
CHAT_PID_FILE="${RUN_DIR}/${CHAT_NAME}.pid"
CHAT_LOG_FILE="${LOG_DIR}/${CHAT_NAME}.log"

PORTAL_NAME="chat_vqa_portal"
PORTAL_DIR="${BASE_DIR}/chat_vqa_portal"
PORTAL_PORT="7862"
PORTAL_HEALTH_PATH="/api/health"
PORTAL_PID_FILE="${RUN_DIR}/${PORTAL_NAME}.pid"
PORTAL_LOG_FILE="${LOG_DIR}/${PORTAL_NAME}.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./webapps_ctl.sh start
  ./webapps_ctl.sh start-portal
  ./webapps_ctl.sh stop-portal
  ./webapps_ctl.sh restart-portal
  ./webapps_ctl.sh stop
  ./webapps_ctl.sh stop-all
  ./webapps_ctl.sh restart
  ./webapps_ctl.sh status
  ./webapps_ctl.sh logs
EOF
}

is_http_up() {
  local port="$1"
  local path="$2"
  curl -fsS --max-time 2 "http://127.0.0.1:${port}${path}" >/dev/null 2>&1
}

is_pid_alive() {
  local pid="$1"
  [[ -n "${pid}" && -d "/proc/${pid}" ]]
}

find_python_pid_by_cwd() {
  local app_dir="$1"
  local pid exe cwd

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    [[ -d "/proc/${pid}" ]] || continue

    exe="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"

    if [[ "${exe}" == "${PYTHON_BIN}" || "${exe}" == "${PYTHON_BIN_DIR}/python"* ]]; then
      if [[ "${cwd}" == "${app_dir}" ]]; then
        echo "${pid}"
        return 0
      fi
    fi
  done < <(pgrep -f "${PYTHON_BIN} app.py" || true)

  return 1
}

wait_until_up() {
  local port="$1"
  local path="$2"
  local retries=20
  local i

  for ((i = 1; i <= retries; i++)); do
    if is_http_up "${port}" "${path}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_one() {
  local name="$1"
  local app_dir="$2"
  local port="$3"
  local health_path="$4"
  local pid_file="$5"
  local log_file="$6"
  local pid

  if is_http_up "${port}" "${health_path}"; then
    echo "[${name}] already running on :${port}"
    return 0
  fi

  pid="$(find_python_pid_by_cwd "${app_dir}" || true)"
  if [[ -n "${pid}" ]] && is_pid_alive "${pid}"; then
    echo "[${name}] process exists (pid=${pid}), waiting for HTTP..."
    if wait_until_up "${port}" "${health_path}"; then
      echo "[${name}] ready on :${port}"
      echo "${pid}" > "${pid_file}"
      return 0
    fi
  fi

  echo "[${name}] starting..."
  (
    cd "${app_dir}"
    nohup "${PYTHON_BIN}" app.py >>"${log_file}" 2>&1 &
    echo $! > "${pid_file}"
  )

  if wait_until_up "${port}" "${health_path}"; then
    echo "[${name}] started on :${port}"
  else
    echo "[${name}] failed to start, check log: ${log_file}" >&2
    return 1
  fi
}

stop_one() {
  local name="$1"
  local app_dir="$2"
  local port="$3"
  local health_path="$4"
  local pid_file="$5"
  local pid=""
  local i

  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -d '[:space:]' < "${pid_file}" || true)"
  fi

  if [[ -z "${pid}" ]]; then
    pid="$(find_python_pid_by_cwd "${app_dir}" || true)"
  fi

  if [[ -z "${pid}" ]] || ! is_pid_alive "${pid}"; then
    if is_http_up "${port}" "${health_path}"; then
      echo "[${name}] warning: HTTP is up but pid not found"
    else
      echo "[${name}] already stopped"
      rm -f "${pid_file}"
      return 0
    fi
  else
    echo "[${name}] stopping pid=${pid}..."
    kill "${pid}" 2>/dev/null || true

    for ((i = 1; i <= 10; i++)); do
      if ! is_pid_alive "${pid}"; then
        break
      fi
      sleep 1
    done

    if is_pid_alive "${pid}"; then
      echo "[${name}] force killing pid=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi

  rm -f "${pid_file}"

  if is_http_up "${port}" "${health_path}"; then
    echo "[${name}] warning: service still responding on :${port}" >&2
    return 1
  fi

  echo "[${name}] stopped"
}

status_one() {
  local name="$1"
  local app_dir="$2"
  local port="$3"
  local health_path="$4"
  local pid_file="$5"
  local pid=""

  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -d '[:space:]' < "${pid_file}" || true)"
  fi
  if [[ -z "${pid}" ]]; then
    pid="$(find_python_pid_by_cwd "${app_dir}" || true)"
  fi

  if is_http_up "${port}" "${health_path}"; then
    if [[ -n "${pid}" ]] && is_pid_alive "${pid}"; then
      echo "[${name}] RUNNING (pid=${pid}, port=${port})"
    else
      echo "[${name}] RUNNING (pid=unknown, port=${port})"
    fi
  else
    if [[ -n "${pid}" ]] && is_pid_alive "${pid}"; then
      echo "[${name}] STARTING/UNHEALTHY (pid=${pid}, port=${port})"
    else
      echo "[${name}] STOPPED (port=${port})"
    fi
  fi
}

show_logs() {
  echo "[trace ] ${TRACE_LOG_FILE}"
  echo "[chat  ] ${CHAT_LOG_FILE}"
  echo "[portal] ${PORTAL_LOG_FILE}"
}

start_all() {
  start_one "${TRACE_NAME}" "${TRACE_DIR}" "${TRACE_PORT}" "${TRACE_HEALTH_PATH}" "${TRACE_PID_FILE}" "${TRACE_LOG_FILE}"
  start_one "${CHAT_NAME}" "${CHAT_DIR}" "${CHAT_PORT}" "${CHAT_HEALTH_PATH}" "${CHAT_PID_FILE}" "${CHAT_LOG_FILE}"
  start_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}" "${PORTAL_LOG_FILE}"
}

start_portal() {
  start_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}" "${PORTAL_LOG_FILE}"
}

stop_all() {
  stop_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}"
  stop_one "${CHAT_NAME}" "${CHAT_DIR}" "${CHAT_PORT}" "${CHAT_HEALTH_PATH}" "${CHAT_PID_FILE}"
  stop_one "${TRACE_NAME}" "${TRACE_DIR}" "${TRACE_PORT}" "${TRACE_HEALTH_PATH}" "${TRACE_PID_FILE}"
}

stop_portal() {
  stop_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}"
}

status_all() {
  status_one "${TRACE_NAME}" "${TRACE_DIR}" "${TRACE_PORT}" "${TRACE_HEALTH_PATH}" "${TRACE_PID_FILE}"
  status_one "${CHAT_NAME}" "${CHAT_DIR}" "${CHAT_PORT}" "${CHAT_HEALTH_PATH}" "${CHAT_PID_FILE}"
  status_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}"
}

cmd="${1:-status}"
case "${cmd}" in
  start)
    start_all
    ;;
  start-portal)
    start_portal
    ;;
  stop)
    stop_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}"
    stop_one "${CHAT_NAME}" "${CHAT_DIR}" "${CHAT_PORT}" "${CHAT_HEALTH_PATH}" "${CHAT_PID_FILE}"
    stop_one "${TRACE_NAME}" "${TRACE_DIR}" "${TRACE_PORT}" "${TRACE_HEALTH_PATH}" "${TRACE_PID_FILE}"
    ;;
  stop-portal)
    stop_portal
    ;;
  stop-all)
    stop_all
    ;;
  restart)
    stop_one "${PORTAL_NAME}" "${PORTAL_DIR}" "${PORTAL_PORT}" "${PORTAL_HEALTH_PATH}" "${PORTAL_PID_FILE}"
    stop_one "${CHAT_NAME}" "${CHAT_DIR}" "${CHAT_PORT}" "${CHAT_HEALTH_PATH}" "${CHAT_PID_FILE}"
    stop_one "${TRACE_NAME}" "${TRACE_DIR}" "${TRACE_PORT}" "${TRACE_HEALTH_PATH}" "${TRACE_PID_FILE}"
    start_all
    ;;
  restart-portal)
    stop_portal
    start_portal
    ;;
  status)
    status_all
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
