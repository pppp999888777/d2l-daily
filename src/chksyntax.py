"""用 jsc 检查页面脚本有没有语法错误（浏览器一遇到语法错误整页的交互都会失效）"""
import json,subprocess,tempfile,os
d=os.path.dirname(os.path.abspath(__file__))+'/'
src=open(d+'minitorch2.js').read()+'\n'+open(d+'page2.js').read()+'\n'+open(d+'quiz.js').read()
p=os.path.join(tempfile.gettempdir(),'d2l_syntax.js')
open(p,'w').write('try{new Function('+json.dumps(src)+');print("syntax ok")}catch(e){print("SYNTAX ERROR",e)}')
print(subprocess.run(['/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc',p],capture_output=True,text=True).stdout.strip())
