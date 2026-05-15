# 依赖库最低版本要求: pandas>=2.0
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件
# 注意: 需安装 schedule 库: pip install schedule

import time
import logging
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("7. 定时任务与数据监控")
print("=" * 60)

# --------------------------------------------------
# 一、配置日志
# --------------------------------------------------

log_path = output_dir / "data_monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(log_path), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logger.info("数据监控系统启动")

# --------------------------------------------------
# 二、模拟数据源
# --------------------------------------------------

def generate_data_snapshot(seed=None):
    if seed is None:
        seed = int(datetime.now().timestamp()) % 10000
    np.random.seed(seed)
    return pd.DataFrame({
        "姓名": [f"员工{i:02d}" for i in range(1, 21)],
        "部门": np.random.choice(["技术部", "市场部", "财务部", "人事部"], 20),
        "月薪": np.random.randint(8000, 25000, 20),
        "绩效评分": np.round(np.random.uniform(60, 100, 20), 1),
    })

def compute_data_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

# --------------------------------------------------
# 三、数据变化检测
# --------------------------------------------------

class DataMonitor:
    def __init__(self):
        self.previous_hash = None
        self.previous_summary = None
        self.change_count = 0

    def check_data(self, df, label=""):
        current_hash = compute_data_hash(df)
        current_summary = {
            "rows": len(df),
            "avg_salary": df["月薪"].mean(),
            "avg_performance": df["绩效评分"].mean(),
            "max_salary": df["月薪"].max(),
            "min_salary": df["月薪"].min(),
        }

        if self.previous_hash is None:
            logger.info(f"[{label}] 首次加载数据，记录数: {len(df)}")
            self.previous_hash = current_hash
            self.previous_summary = current_summary
            return "INITIAL"

        if current_hash != self.previous_hash:
            self.change_count += 1
            changes = []
            for key in current_summary:
                old_val = self.previous_summary[key]
                new_val = current_summary[key]
                if old_val != new_val:
                    if isinstance(old_val, float):
                        changes.append(f"  {key}: {old_val:.2f} -> {new_val:.2f}")
                    else:
                        changes.append(f"  {key}: {old_val} -> {new_val}")

            logger.warning(f"[{label}] 检测到数据变化 (第{self.change_count}次)!")
            for change in changes:
                logger.warning(change)

            self.previous_hash = current_hash
            self.previous_summary = current_summary
            return "CHANGED"

        logger.info(f"[{label}] 数据无变化")
        return "UNCHANGED"

monitor = DataMonitor()

df_v1 = generate_data_snapshot(seed=42)
result = monitor.check_data(df_v1, "V1")
print(f"\n[检测] V1: {result}")

df_v2 = generate_data_snapshot(seed=42)
result = monitor.check_data(df_v2, "V2(相同)")
print(f"[检测] V2(相同数据): {result}")

df_v3 = generate_data_snapshot(seed=100)
result = monitor.check_data(df_v3, "V3(不同)")
print(f"[检测] V3(不同数据): {result}")

# --------------------------------------------------
# 四、使用 schedule 库设置定时任务
# --------------------------------------------------

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    print("\n[提示] schedule 库未安装，将使用自定义定时逻辑演示")
    print("  安装命令: pip install schedule")

if HAS_SCHEDULE:
    print("\n--- schedule 库定时任务演示 ---")

    def job_data_check():
        logger.info("定时任务: 执行数据检查")
        df = generate_data_snapshot()
        result = monitor.check_data(df, "定时检查")
        if result == "CHANGED":
            logger.warning("数据已变化，建议发送告警通知")

    def job_daily_report():
        logger.info("定时任务: 生成日报")
        df = generate_data_snapshot()
        report_path = output_dir / f"日报_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_path, index=False, encoding="utf-8-sig")
        logger.info(f"日报已保存: {report_path}")

    schedule.every(10).minutes.do(job_data_check)
    schedule.every().day.at("09:00").do(job_daily_report)
    schedule.every().monday.at("10:00").do(job_daily_report)

    print("已注册定时任务:")
    for job in schedule.get_jobs():
        print(f"  - {job}")

# --------------------------------------------------
# 五、自定义定时执行逻辑
# --------------------------------------------------

print("\n--- 自定义定时执行演示 ---")

def run_scheduled_tasks(task_list, max_iterations=3, interval_seconds=1):
    for iteration in range(max_iterations):
        logger.info(f"=== 第 {iteration + 1} 次执行 ===")
        for task_name, task_func in task_list:
            try:
                logger.info(f"执行任务: {task_name}")
                task_func()
            except Exception as e:
                logger.error(f"任务 {task_name} 执行失败: {e}")
        if iteration < max_iterations - 1:
            time.sleep(interval_seconds)

def task_check_data():
    df = generate_data_snapshot()
    result = monitor.check_data(df, "自定义定时")
    print(f"  [数据检查] 结果: {result}")

def task_summary():
    df = generate_data_snapshot()
    summary = f"记录数: {len(df)}, 平均薪资: {df['月薪'].mean():,.0f}"
    print(f"  [数据摘要] {summary}")
    logger.info(f"数据摘要: {summary}")

tasks = [
    ("数据变化检查", task_check_data),
    ("数据摘要生成", task_summary),
]

run_scheduled_tasks(tasks, max_iterations=3, interval_seconds=1)

# --------------------------------------------------
# 六、后台进程概念说明
# --------------------------------------------------

print("\n--- 后台运行方式说明 ---")
print("1. nohup 方式 (Linux/Mac):")
print("   nohup python 7_定时任务与数据监控.py > monitor.log 2>&1 &")
print()
print("2. Windows 服务方式:")
print("   python 7_定时任务与数据监控.py  # 在后台任务中运行")
print()
print("3. systemd 方式 (Linux):")
print("   创建 /etc/systemd/system/data-monitor.service")
print("   [Unit]")
print("   Description=Data Monitor Service")
print("   [Service]")
print("   ExecStart=/usr/bin/python3 7_定时任务与数据监控.py")
print("   Restart=always")
print("   [Install]")
print("   WantedBy=multi-user.target")
print()
print("4. Docker 方式:")
print("   docker run -d --name data-monitor python:3.11 python app.py")
print()
print("5. cron 定时 (Linux):")
print("   0 9 * * 1-5 /usr/bin/python3 /path/to/script.py")

# --------------------------------------------------
# 七、监控结果汇总
# --------------------------------------------------

print(f"\n--- 监控统计 ---")
print(f"数据变化检测次数: {monitor.change_count}")
print(f"日志文件: {log_path}")

logger.info("数据监控演示完成")

print("\n" + "=" * 60)
print("定时任务与数据监控 - 完成")
print("=" * 60)
