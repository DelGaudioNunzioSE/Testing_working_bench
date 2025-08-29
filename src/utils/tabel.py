def render_summary_table(
    human_code="10,000",
    llms_code="199,988",
    num_llms_desc="10  LLMs: <em>(GPT-4o-mini, GPT-o3-mini, Qwen-2.5-Coder-32B, Claude-3.5-Haiku, DeepSeek-R1, DeepSeek-V3, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro, Llama-3.3-70B)</em>",
    diversity_desc="6 different source models: <em>(GPT, CodeQwen, Claude, DeepSeek, Gemini, Llama)</em>",
    use_period="2024–2025",
    languages_desc="<em>(HTML, JavaScript, Java, C, Python, Ruby, Go, C#, C++, PHP)</em>",
    code_types="unspecified",
    code_size="1<sup>st</sup> percentile: 639 words, 3<sup>rd</sup> percentile: 804 words",
    code_context="open-source",
    prompts="Provided (one)",
    source_human="GitHub",
    code_quality="Statistical alignment",
    reliability="Medium, <em>(no precise references are provided regarding the data source)</em>",
    note="Evaluation Summary: CodeMirage Dataset",
    left_width="50%", right_width="50%"
):
    CSS = f"""
    <style>
    .cm-table {{ width:100%; border-collapse:collapse; table-layout: fixed; }}
    .cm-table tr, .cm-table td, .cm-table th {{ border:1px solid #444; padding:8px; vertical-align:top; }}
    th.section {{ background:#111; color:#fff; text-align:center; }}
    .cm-note {{ text-align:center; margin-top:6px; font-size:13px; opacity:.8; }}
    .cm-col-left {{ width: {left_width}; }}
    .cm-col-right {{ width: {right_width}; }}
    </style>
    """



# list of tuple (title,([(description, value),(description, value),...]))
    SECTIONS = [
        ("Total number of code samples available", [
            ("(a) Total number of human code", human_code),
            ("(b) Total number of LLMs code", llms_code),
        ]),
        ("LLMs used to generate synthetic code", [
            ("(a) Number of LLMs", num_llms_desc),
            ("(b) Diversity among LLMs", diversity_desc),
            ("(c) Actual degree of use in contemporary times", use_period),
        ]),
        ("Code diversity", [
            ("(a) Different programming languages", languages_desc),
            ("(b) Different types of code", code_types),
            ("(c) Code size", code_size),
            ("(d) Code context", code_context),
        ]),
        ("Validity information", [
            ("(a) Generation prompts", prompts),
            ("(b) Source of human-written code", source_human),
            ("(c) Code quality", code_quality),
            ("(d) Paper reliability perception", reliability),
        ]),
    ]

    def render_rows(sections):
        rows = []
        for title, items in sections:
            rows.append(f'<tr><th class="section" colspan="2">{title}</th></tr>')
            for left, right in items:
                rows.append(f'<tr><td class="cm-col-left"><b>{left}</b></td><td class="cm-col-right">{right}</td></tr>')
        return "\n".join(rows)

    html = f"""
    {CSS}
    <table class="cm-table">
    {render_rows(SECTIONS)}
    </table>
    <div class="cm-note">{note}</div>
    """
    return html
