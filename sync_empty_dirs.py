import os
import subprocess
from pathlib import Path

# ⚠️ 重点检查：黑名单目录（防暴雷机制）
# 这里的目录及其所有子目录将被彻底跳过，不会生成占位文件。
# 请务必根据你的实际项目情况（例如数据集路径、虚拟环境等）修改此列表！
BLACKLIST_DIRS = {
    '.git', 
    'node_modules', 
    'venv', 
    'env', 
    '.env', 
    '__pycache__', 
    '.idea', 
    '.vscode',
    '__MACOSX'
}

def is_tracked_by_git(dir_path):
    """
    检查目录下是否有已经被 Git 追踪的文件
    """
    try:
        # git ls-files 会列出该目录下所有被追踪的文件
        # 如果返回结果为空字符串，说明没有任何追踪文件
        result = subprocess.run(
            ['git', 'ls-files', str(dir_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"执行 Git 命令失败: {e}")
        return False

def process_directory(root_path):
    root_path = Path(root_path).resolve()
    
    # 基础校验：确保在 Git 仓库内
    if not (root_path / '.git').exists() and not is_tracked_by_git(root_path):
        print(f"错误: {root_path} 似乎不是一个 Git 仓库的根目录。请在项目根目录运行。")
        return

    added_count = 0

    # 使用 bottom-up (自底向上) 遍历
    # 这样可以先在最深层的空目录创建占位文件，上层目录就会自动被 Git 识别为"有追踪文件"而跳过重复创建
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        current_dir = Path(dirpath)
        
        # 检查当前路径是否触碰黑名单
        if any(part in BLACKLIST_DIRS for part in current_dir.parts):
            continue

        rel_path = current_dir.relative_to(root_path)
        if str(rel_path) == '.':
            continue

        # 如果目录下没有任何被追踪的文件
        if not is_tracked_by_git(current_dir):
            placeholder_path = current_dir / '.gitignore'
            
            if placeholder_path.exists():
                continue
            
            print(f"检测到未追踪/被忽略的目录: {rel_path}，正在添加占位文件...")
            
            try:
                # 写入局部 .gitignore 规则：忽略该目录下所有文件，但保留此 .gitignore 自身
                with open(placeholder_path, 'w', encoding='utf-8') as f:
                    f.write("# Ignore everything in this directory\n*\n# Except this file\n!.gitignore\n")
                
                # 强制添加到 Git 的暂存区，无视根目录的 .gitignore 拦截
                subprocess.run(
                    ['git', 'add', '-f', str(placeholder_path)],
                    check=True,
                    capture_output=True
                )
                added_count += 1
            except Exception as e:
                print(f"处理目录 {rel_path} 时出错: {e}")

    print(f"\n执行完毕。共为 {added_count} 个空/被忽略的路径添加了占位文件。")

if __name__ == '__main__':
    # 默认处理运行该脚本的当前目录（建议在项目根目录下执行）
    process_directory('.')