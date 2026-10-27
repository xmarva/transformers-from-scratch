import gradio as gr

EXAMPLES = ["I think, therefore I am.", "Do you dream in color?", "Nobody understands me.",
            "Memory is the diary we all carry about with us."]
ABOUT = """# perevod
English to Russian with a small Transformer built from scratch in the course
[Transformers: from scratch to production][course]. It was trained on short everyday
sentences; its [model card][card] says where it fails.

[course]: https://github.com/xmarva/transformers-from-scratch
[card]: https://huggingface.co/xmarva/perevod-en-ru"""


def build_demo(app):
    async def translate(text):
        return await app.state.batcher.translate(text)  # through the same queue as the API

    with gr.Blocks(title="perevod") as demo:
        gr.Markdown(ABOUT)
        source = gr.Textbox(label="English", lines=3)
        target = gr.Textbox(label="Russian", lines=3)
        button = gr.Button("Translate", variant="primary")
        button.click(translate, source, target)
        gr.Examples(EXAMPLES, source)
    return demo
