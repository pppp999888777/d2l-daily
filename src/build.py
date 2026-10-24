"""把每日课和调试台组装成可以直接放在 GitHub Pages 上的网页。

用法：python3 src/build.py
输出：daily.html（每日课）、playground.html（调试台）、index.html（首页），以及 ../动手学深度学习每日课.html（claude.ai 版本）
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = lambda *a: os.path.join(HERE, *a)
MT_PARTS = ['mt2_core.js', 'mt2_ops.js', 'mt2_lib.js', 'mt2_mods.js', 'mt2_lang.js', 'mt2_nn.js', 'mt2_vision.js', 'mt2_eval.js']

def read(name):
    with open(P(name), encoding='utf-8') as f:
        return f.read()

def artifact_daily():
    mt = ''.join(read(n) for n in MT_PARTS)
    with open(P('minitorch2.js'), 'w', encoding='utf-8') as f:
        f.write(mt)
    data = '<script type="text/plain" id="fmData">' + read('fmnist_b64.txt').strip() + '</script>\n'
    # 时光机器全文（第 8 章用），原样放进去；页面启动时登记到模拟文件系统的 ../data/timemachine.txt
    data += '<script type="text/plain" id="tmData">' + read('timemachine.txt') + '</script>\n'
    data += '<script type="text/plain" id="afData">' + read('airfoil_self_noise.dat') + '</script>\n'
    return read('page_head_full.html') + read('page_body2.html') + '\n' + data + '<script>\n' + mt + '\n' + read('page2.js') + '\n' + read('quiz.js') + '</script>\n'

SKELETON = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,-apple-system,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
'''

def standalone(fragment):
    # claude.ai 版本没有 <!doctype>/<head>/<body>，单独放到网上时补上
    return SKELETON + fragment + '\n</body>\n</html>\n'

def main():
    daily = artifact_daily()
    with open(os.path.join(os.path.dirname(ROOT), '动手学深度学习每日课.html'), 'w', encoding='utf-8') as f:
        f.write(daily)
    with open(os.path.join(ROOT, 'daily.html'), 'w', encoding='utf-8') as f:
        f.write(standalone(daily))
    with open(os.path.join(ROOT, 'playground.html'), 'w', encoding='utf-8') as f:
        f.write(standalone(read('playground_artifact.html')))
    with open(os.path.join(ROOT, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(standalone(read('index_fragment.html')))
    print('built daily.html, playground.html, index.html')

if __name__ == '__main__':
    main()
