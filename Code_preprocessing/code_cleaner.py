import re


def comment_remover(code, language="Python"):
    if code is None:
        print('Input code is None')
        return



    if language == "Python":
        pattern = re.compile(     
            r'""".*?"""'                  # triple double-quoted string
            r"|'''.*?'''"                 # triple single-quoted string
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language in ("C", "CPP", "Go"):
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string (char literal or escaped)
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|`[^`]*`',                  # raw string in Go using backticks
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language in ("CSharp", "C#"):
        pattern = re.compile(
            r'//.*?$'                     # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|@\"(?:[^\"]|\"\")*\"',     # verbatim string @"..." ("" is escape)
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "Java":
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

    elif language == "JavaScript":
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

    elif language == "Ruby":
        # =begin / =end must be at the beginning of the line (no spaces before)
        pattern = re.compile(
            r'^=begin.*?^=end$'           # multi-line block comment
            r'|#.*?$',                    # single-line comment
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "PHP":
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "HTML":
        pattern = re.compile(
            r'<!--.*?-->', re.DOTALL # HTML comment <!-- ... -->
        )  
        code = re.sub(pattern, '', code)

    else:
        print(f"{language} is not supported")

    if code is None:
        raise ValueError('Output code is None')
    


    # FOR EVRY_LANGUAGE
    code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)
    
    return code