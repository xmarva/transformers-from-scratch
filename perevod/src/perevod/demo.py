import gradio as gr

EXAMPLES = ["My brother lives in a small town by the sea.",
            "Every morning I drink tea and read the news.",
            "The cat is sleeping on the windowsill.", "Tom is afraid of the dark."]
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
