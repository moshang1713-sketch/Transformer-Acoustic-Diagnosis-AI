import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    # 处理 PyInstaller 打包后的临时路径
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)

if __name__ == "__main__":
    # 强制指定运行 st_app.py
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("st_app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())