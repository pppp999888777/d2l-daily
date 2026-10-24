ObjC.import('Foundation');
var src=$.NSString.stringWithContentsOfFileEncodingError('minitorch2.js',4,null).js;
try{new Function(src);'SYNTAX_OK'}catch(e){'ERR '+e}
