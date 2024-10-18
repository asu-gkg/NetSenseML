#!/bin/bash

# 检查是否提供了端口号
if [ -z "$1" ]; then
  echo "请提供端口号。例如：./kill_port.sh 8000"
  exit 1
fi

PORT=$1

# 查找占用端口的进程ID
PID=$(sudo lsof -t -i :$PORT)

# 如果没有进程占用该端口
if [ -z "$PID" ]; then
  echo "端口 $PORT 没有被任何进程占用。"
else
  # 显示占用端口的进程信息
  echo "端口 $PORT 被以下进程占用："
  sudo lsof -i :$PORT
  
  # 终止进程
  echo "正在终止占用端口 $PORT 的进程: $PID"
  sudo kill -9 $PID
  
  # 检查进程是否成功终止
  if [ $? -eq 0 ]; then
    echo "进程 $PID 已成功终止，端口 $PORT 现在空闲。"
  else
    echo "无法终止进程 $PID，请检查进程权限。"
  fi
fi