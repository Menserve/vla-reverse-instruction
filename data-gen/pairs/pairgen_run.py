"""段 5 の教師の組の生成（container の中）。s2/pairgen.py をそのまま走らせる（種の土台と組の型の表だけ /a の s5 の assets から読む）。
引数の形は pairgen.py と同じ: pairgen_run.py <出力 dir> <組の型:始め:数> ...
"""
import runpy

runpy.run_path("/s2/pairgen.py", run_name="__main__")
