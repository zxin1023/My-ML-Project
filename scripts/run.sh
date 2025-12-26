#!/bin/bash
# 使用方法：bash run_exp.sh configs/exp1.yaml

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG=$1
EXP_NAME=$(grep 'exp_name' $CONFIG | awk '{print $2}' | tr -d '"')
TMUX_SESSION="exp_${EXP_NAME}"
LOG_FILE="experiments/${EXP_NAME}/run.log"

# 创建日志目录
mkdir -p experiments/${EXP_NAME}

# 在tmux会话中运行实验，并重定向日志
tmux new -d -s $TMUX_SESSION \
  "python src/train.py --config $CONFIG > $LOG_FILE 2>&1; exec bash"

echo "实验已启动："
echo "  TMUX会话名: $TMUX_SESSION"
echo "  日志文件: $LOG_FILE"
echo "  查看会话: tmux attach -t $TMUX_SESSION"
echo "  查看日志: tail -f $LOG_FILE"