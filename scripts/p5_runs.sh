#!/usr/bin/env bash
# P5 的全部实验运行清单（方案第五、六、八节），按阶段顺序执行；每条命令的输出追加到 data/logs/p5_runs.log。
# 用法（在项目根目录）：
#     bash scripts/p5_runs.sh ablation      # 11 次消融（v1 dev）
#     bash scripts/p5_runs.sh v0            # v0 佐证：闭卷 / 固定 RAG / Agent（dev + test 共 120 题）
#     bash scripts/p5_runs.sh test          # v1 test 一次性确认：闭卷 / 固定 RAG / 固定 GraphRAG / Agent × 2
# 已有同名结果目录的运行会跳过（中断后可以续跑；LLM 结果另有缓存）。test 阶段只在配置冻结后执行一次。
set -u
CONC="${P5_CONCURRENCY:-8}"
LOG=data/logs/p5_runs.log
mkdir -p data/logs

run() {  # run <名称> <run_qa.py 的其余参数...>
  local name="$1"; shift
  if ls -d eval/results/*_"$name" >/dev/null 2>&1; then
    echo "[skip] $name 已有结果" | tee -a "$LOG"; return 0
  fi
  echo "[start] $name $(date '+%F %T')" | tee -a "$LOG"
  uv run python scripts/run_qa.py --name "$name" --concurrency "$CONC" "$@" >>"$LOG" 2>&1
  echo "[done] $name exit=$? $(date '+%F %T')" | tee -a "$LOG"
}

case "${1:-}" in
  ablation)
    # 主要变体各两次采样
    for s in 0 1; do
      run p5_abl_nograph_s$s --system agent --sample $s --no-graph
      run p5_abl_nosum_s$s   --system agent --sample $s --drop-tools search_summaries,read_summary
      run p5_abl_raw_s$s     --system agent --sample $s --act-memory raw
    done
    # 其余变体各一次
    run p5_abl_nogq_s0   --system agent --sample 0 --drop-tools graph_query
    run p5_abl_person_s0 --system agent --sample 0 --graph-scope person
    run p5_abl_low_s0    --system agent --sample 0 --max-rounds 3 --max-tool-calls 6 --max-tokens 20000
    run p5_abl_high_s0   --system agent --sample 0 --max-rounds 10 --max-tool-calls 20 --max-tokens 60000
    run p5_abl_think_s0  --system agent --sample 0 --thinking
    ;;
  v0)
    for sys in closed rag agent; do
      run p5_v0_$sys --system $sys --dataset asoiaf_v0 --split all --confirm-test
    done
    ;;
  test)
    run p5_test_closed   --system closed   --split test --confirm-test
    run p5_test_rag      --system rag      --split test --confirm-test
    run p5_test_graphrag --system graphrag --split test --confirm-test
    run p5_test_agent_s0 --system agent    --split test --confirm-test --sample 0
    run p5_test_agent_s1 --system agent    --split test --confirm-test --sample 1
    ;;
  *)
    echo "用法：bash scripts/p5_runs.sh ablation|v0|test" >&2; exit 2 ;;
esac
