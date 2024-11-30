import ollama
from rich import print
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    DownloadColumn,
    Progress,
    TextColumn
)

CHAT_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"


def verify_models_pulled():
    pulled_models = ollama.list().models

    if any(CHAT_MODEL_NAME in model.model for model in pulled_models):
        print(
            f"[green]Chat model [blue]{CHAT_MODEL_NAME} [green]already pulled!")
    else:
        print(
            f"[yellow]Chat model [blue]{CHAT_MODEL_NAME} [yellow]not pulled. Pulling...")
        _streaming_pull(CHAT_MODEL_NAME)
        print("[green]Done!")

    if any(EMBEDDING_MODEL_NAME in model.model for model in pulled_models):
        print(
            f"[green]Embedding model [blue]{EMBEDDING_MODEL_NAME} [green]already pulled!")
    else:
        print(
            f"[yellow]Embedding model [blue]{EMBEDDING_MODEL_NAME} [yellow]not pulled. Pulling...")
        _streaming_pull(EMBEDDING_MODEL_NAME)
        print("[green]Done!")


def update_models():
    _streaming_pull(f"{CHAT_MODEL_NAME}:latest")
    _streaming_pull(f"{EMBEDDING_MODEL_NAME}:latest")


def _streaming_pull(model_name: str):
    '''
    Pulls an Ollama model and provides status updates
    '''

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("•"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    with progress_bar:

        current_digest = None
        task = None

        for progress in ollama.pull(model_name, stream=True):

            if (progress.digest is not None and
                progress.digest != current_digest):

                current_digest = progress.digest
                task = progress_bar.add_task(f"Pulling [blue]{current_digest[7:19]}")

            if task is not None:
                progress_bar.update(task, total=progress.total,
                                    completed=progress.completed)


if __name__ == "__main__":
    verify_models_pulled()
    update_models()
