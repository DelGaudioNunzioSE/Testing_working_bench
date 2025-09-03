import re


def comment_remover(code, language="python"):
    if code is None:
        print('Input code is None')
        return

    language = language.lower()

    if language == "python":
        pattern = re.compile(     
            r'""".*?"""'                  # triple double-quoted string
            r"|'''.*?'''"                 # triple single-quoted string
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language in ("c", "cpp", "c++", "go"):  #cpp is an other way to say c++
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string (char literal or escaped)
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|`[^`]*`',                  # raw string in Go using backticks
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language in ("csharp", "c#"):
        pattern = re.compile(
            r'//.*?$'                     # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|@\"(?:[^\"]|\"\")*\"',     # verbatim string @"..." ("" is escape)
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "java":
        pattern = re.compile(
            r'/\*\*.*?\*/'                # Javadoc comment /** ... */
            r'|//.*?$'                    # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|"""(?:.|\n)*?"""',         # text block (Java 15+)
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "javascript":
        pattern = re.compile(
            r'/\*\*.*?\*/'                # JSDoc comment /** ... */
            r'|//.*?$'                    # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|`(?:\\.|[^\\`])*`',        # template literal
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "ruby":
        # =begin / =end must be at the beginning of the line (no spaces before)
        pattern = re.compile(
            r'^=begin.*?^=end$'           # multi-line block comment
            r'|#.*?$',                    # single-line comment
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "php":
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "html":
        pattern = re.compile(
            r'<!--.*?-->', re.DOTALL # HTML comment <!-- ... -->
        )  
        code = re.sub(pattern, '', code)

    else:
        print(f"{language} is not supported")

    if code is None:
        raise ValueError('Output code is None')
    
    
    return code




def newline_remover(code):

    code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

    return code




def import_remover(code, language="python"):
    if code is None:
        return
    language = language.lower()

    if language == "python":
        # singola riga + from ... import (...) multiline
        code = re.sub(r'(?ms)^\s*(?:from\s+[A-Za-z_][\w\.]*\s+import\s*\([\s\S]*?\)\s*'
                      r'|from\s+[A-Za-z_][\w\.]*\s+import\s+[^\n#]+'
                      r'|import\s+[^\n#]+)\s*$', '', code)

    elif language in ("c", "cpp", "c++"):
        code = re.sub(r'(?m)^\s*#\s*include\s*[<"].*[>"].*$', '', code)

    elif language in ("csharp", "c#"):
        code = re.sub(r'(?m)^\s*using\s+[\w\.]+(?:\s*=\s*[\w\.]+)?\s*;\s*$', '', code)

    elif language == "java":
        code = re.sub(r'(?m)^\s*import\s+(?:static\s+)?[\w\.]+(?:\.\*)?\s*;\s*$', '', code)

    elif language in ("javascript", "typescript", "ts", "jsx", "tsx"):
        # import ...;  + require(...) varianti comuni
        code = re.sub(r'(?m)^\s*import\s+[^;]*;?\s*$', '', code)
        code = re.sub(r'(?m)^\s*(?:const|let|var)\s+[\w$]+\s*=\s*require\([^)]*\)\s*;?\s*$', '', code)
        code = re.sub(r'(?m)^\s*require\([^)]*\)\s*;?\s*$', '', code)

    elif language == "ruby":
        code = re.sub(r'(?m)^\s*require(?:_relative)?\s+.+$', '', code)

    elif language == "php":
        code = re.sub(r'(?m)^\s*use\s+[^;]+;\s*$', '', code)
        code = re.sub(r'(?m)^\s*(?:require|require_once|include|include_once)\s*\([^)]*\)\s*;\s*$', '', code)

    elif language == "go":
        # import (...) blocco + singola riga
        code = re.sub(r'(?ms)^\s*import\s*\(\s*[\s\S]*?\)\s*', '', code)
        code = re.sub(r'(?m)^\s*import\s+["\'][^"\']+["\']\s*$', '', code)

    elif language == "html":
        pass  # nessun import da rimuovere

    else:
        print(f"{language} is not supported")

    return code
