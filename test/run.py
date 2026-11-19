"""兼容旧命令；主入口在 scripts/run.py。"""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 使用可导入模块，确保 Windows multiprocessing spawn 能找到工作函数。
from scripts.run import main


if __name__ == "__main__":
    main()
