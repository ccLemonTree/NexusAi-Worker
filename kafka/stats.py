"""
Kafka 消费/生产统计模块
- 按分钟统计消费和生产的消息条数
- 定时写入本地 CSV 文件
- 每个服务实例独立记录
"""
import csv
import os
import asyncio
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from threading import Lock

from tools.logger_tools import Kafka_Consumer_logger as logger


class KafkaStatsCollector:
    """按分钟统计 Kafka 消费/生产条数，定时写入 CSV"""

    def __init__(self, stats_dir: str = "/logs"):
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # 主机名（区分不同服务实例）
        self.hostname = os.getenv("HOSTNAME", "unknown")

        # 当前分钟的计数器：{minute_key: {"consumed": count, "produced": count}}
        self.current_minute = self._current_minute_key()
        self.consumed_count = 0
        self.produced_count = 0
        self.lock = Lock()

        # CSV 文件路径：kafka_stats_{hostname}_{date}.csv
        self.csv_path = self._get_csv_path()
        self._init_csv()

        logger.info(f"统计模块已启动 hostname={self.hostname} csv={self.csv_path}")

    def _current_minute_key(self) -> str:
        """返回当前分钟的 key，格式 'YYYY-MM-DD HH:MM'"""
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _get_csv_path(self) -> Path:
        """生成 CSV 文件路径：kafka_stats_{hostname}_{date}.csv"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.stats_dir / f"kafka_stats_{self.hostname}_{date_str}.csv"

    def _init_csv(self):
        """如果 CSV 文件不存在，创建并写入表头"""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "consumed", "produced"])
            logger.info(f"创建统计文件: {self.csv_path}")

    def record_consumed(self):
        """记录一条消费"""
        with self.lock:
            self.consumed_count += 1

    def record_produced(self):
        """记录一条生产"""
        with self.lock:
            self.produced_count += 1

    def _flush_to_csv(self):
        """将当前分钟的统计写入 CSV"""
        with self.lock:
            if self.consumed_count == 0 and self.produced_count == 0:
                return  # 没有数据，跳过

            # 检查日期是否变化（跨天了）
            new_csv = self._get_csv_path()
            if new_csv != self.csv_path:
                self.csv_path = new_csv
                self._init_csv()

            # 写入 CSV
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_minute,
                    self.consumed_count,
                    self.produced_count,
                ])

            logger.info(
                f"统计写入 {self.current_minute}  consumed={self.consumed_count}  produced={self.produced_count}"
            )

            # 重置计数器
            self.consumed_count = 0
            self.produced_count = 0

    async def run_periodic_flush(self):
        """后台任务：每分钟检查一次，如果分钟变化则写入 CSV"""
        while True:
            await asyncio.sleep(10)  # 每 10 秒检查一次
            current = self._current_minute_key()
            if current != self.current_minute:
                # 分钟变化，写入上一分钟的数据
                self._flush_to_csv()
                self.current_minute = current


# 全局单例
_stats_collector: KafkaStatsCollector = None


def get_stats_collector() -> KafkaStatsCollector:
    """获取全局统计收集器单例"""
    global _stats_collector
    if _stats_collector is None:
        stats_dir = os.getenv("KAFKA_STATS_DIR", "/logs")
        _stats_collector = KafkaStatsCollector(stats_dir)
    return _stats_collector


def start_stats_task():
    """启动统计后台任务（在主事件循环中调用）"""
    collector = get_stats_collector()
    asyncio.create_task(collector.run_periodic_flush())
    logger.info("统计后台任务已启动")
