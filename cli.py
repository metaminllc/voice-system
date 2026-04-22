"""Command-line interface for the Voice Distillation system.

Commands:

    ingest           Ingest a corpus file (novel / essay / review / diary).
    add              Add a dialogue fragment (optionally linked to corpus files).
    list-dialogues   Show all captured dialogue fragments.
    rebut            Summon her rebuttal to a draft passage.
    retrieve         Peek at what the retriever returns for a query.
    web              Start the web interface.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from voice.dialogue import add_dialogue, list_dialogues
from voice.generate import generate_rebuttal
from voice.ingest import ingest
from voice.retrieve import retrieve

app = typer.Typer(
    help="Voice Distillation — distill her voice, call on her to push back.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("ingest")
def cli_ingest(
    file_path: str = typer.Argument(..., help="Path to the file to ingest."),
    type_: str = typer.Option(
        ..., "--type", "-t",
        help="Corpus type: novel | essay | review | diary.",
    ),
    date: Optional[str] = typer.Option(None, "--date", help="Date (YYYY-MM-DD or YYYY)."),
    tags: Optional[str] = typer.Option(
        None, "--tags",
        help="Comma-separated topic tags, e.g. 启蒙理性,女性",
    ),
    stance: Optional[str] = typer.Option(
        None, "--stance",
        help="Her relationship to the content: affirm | resist | ambivalent | expository",
    ),
    stance_note: Optional[str] = typer.Option(
        None, "--stance-note",
        help="Free-text annotation of stance, e.g. "
             "'对卡西尔更同情，但认为海德格尔提出了真正的问题'",
    ),
    sentiment: Optional[str] = typer.Option(
        None, "--sentiment",
        help="(legacy alias for --stance; prefer --stance)",
    ),
    book: Optional[str] = typer.Option(None, "--book", help="Book title (for reviews)."),
    judgment: Optional[str] = typer.Option(
        None, "--judgment",
        help="Review judgment direction: affirm | resist | ambivalent | expository",
    ),
):
    """Ingest a corpus file into ChromaDB."""
    meta = {}
    if date:
        meta["date"] = date
    if tags:
        meta["topic_tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if stance:
        meta["stance"] = stance
    if stance_note:
        meta["stance_note"] = stance_note
    if sentiment:
        meta["sentiment"] = sentiment
    if book:
        meta["book"] = book
    if judgment:
        meta["judgment"] = judgment

    with console.status(f"Ingesting [cyan]{file_path}[/cyan] as [magenta]{type_}[/magenta]..."):
        try:
            n = ingest(file_path, type_, meta)
        except FileNotFoundError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(code=1)
        except ValueError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(code=1)
    console.print(
        f"[green]✓[/green] Ingested [bold]{n}[/bold] chunk(s) from [cyan]{file_path}[/cyan]"
    )


@app.command("add")
def cli_add(
    quote: str = typer.Argument(..., help="The quote or paraphrase."),
    context: Optional[str] = typer.Option(None, "--context", help="Context for the quote."),
    note: Optional[str] = typer.Option(None, "--note", help="Your own annotation."),
    confidence: str = typer.Option(
        "paraphrase", "--confidence",
        help="exact | paraphrase | inference",
    ),
    link: Optional[List[str]] = typer.Option(
        None, "--link",
        help="Corpus file stem this quote is semantically bound to. "
             "Pass multiple times to link against several, e.g. "
             "--link lotr_review_20080116 --link magic_mountain_essay. "
             "Extensions like .md/.txt are stripped automatically.",
    ),
):
    """Capture a dialogue fragment (ad-hoc), optionally linked to corpus files."""
    try:
        record = add_dialogue(
            quote=quote,
            context=context,
            your_note=note,
            confidence=confidence,
            linked_sources=link,
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]✓[/green] Saved dialogue fragment [dim]{record['id']}[/dim]"
    )
    title = f"confidence: {confidence}"
    if record.get("linked_sources"):
        title += f"  |  linked: {', '.join(record['linked_sources'])}"
    console.print(Panel(quote, title=title, border_style="blue"))


@app.command("list-dialogues")
def cli_list_dialogues():
    """List every dialogue fragment captured so far."""
    records = list_dialogues()
    if not records:
        console.print("[yellow]No dialogue fragments yet.[/yellow]")
        return
    table = Table(title=f"Dialogue fragments ({len(records)})")
    table.add_column("Quote", style="cyan", overflow="fold", max_width=60)
    table.add_column("Context", overflow="fold", max_width=40)
    table.add_column("Confidence", style="magenta")
    table.add_column("Linked", style="green", overflow="fold", max_width=30)
    table.add_column("Created", style="dim")
    for r in records:
        links = r.get("linked_sources") or []
        if isinstance(links, str):
            links_str = links
        else:
            links_str = ", ".join(links)
        table.add_row(
            r.get("quote", ""),
            r.get("context") or "",
            r.get("confidence", ""),
            links_str,
            (r.get("created_at") or "")[:19],
        )
    console.print(table)


@app.command("rebut")
def cli_rebut(
    draft: str = typer.Argument(..., help="Your draft passage."),
    mode: str = typer.Option(
        "pressure", "--mode", "-m",
        help="pressure | dialogue | annotation",
    ),
    n: int = typer.Option(6, "--n", help="Number of retrieved chunks."),
):
    """Summon her rebuttal."""
    with console.status("Retrieving relevant corpus..."):
        chunks = retrieve(draft, n_results=n)

    if chunks:
        console.print(f"[dim]Retrieved {len(chunks)} chunk(s):[/dim]")
        for c in chunks:
            label = f"{c.source_type}:{c.source_file}"
            if c.confidence:
                label += f" ({c.confidence})"
            reason = c.metadata.get("_retrieval_reason")
            dist_str = "linked" if c.distance == float("inf") else f"dist={c.distance:.3f}"
            suffix = f" [yellow]{reason}[/yellow]" if reason else ""
            console.print(f"  • [cyan]{label}[/cyan]  [dim]{dist_str}[/dim]{suffix}")
    else:
        console.print("[yellow]No retrieved chunks — generation will lean on the persona file only.[/yellow]")

    with console.status(f"Generating rebuttal in [magenta]{mode}[/magenta] mode..."):
        try:
            rebuttal = generate_rebuttal(draft, chunks, mode=mode)
        except ValueError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(code=1)
        except RuntimeError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(code=1)

    console.print(Panel(Markdown(rebuttal), title=f"她的反驳 [{mode}]", border_style="red"))


@app.command("web")
def cli_web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
):
    """Start the FastAPI web interface at http://HOST:PORT."""
    import uvicorn
    console.print(
        f"[green]→[/green] Voice web UI starting at "
        f"[cyan]http://{host}:{port}[/cyan]  [dim](Ctrl-C to stop)[/dim]"
    )
    uvicorn.run("app:app", host=host, port=port, reload=reload)


@app.command("retrieve")
def cli_retrieve(
    query: str = typer.Argument(..., help="Query string."),
    n: int = typer.Option(5, "--n", help="Number of chunks to return."),
):
    """Inspect raw retriever output (for debugging)."""
    chunks = retrieve(query, n_results=n)
    if not chunks:
        console.print("[yellow]Nothing retrieved — is your corpus empty?[/yellow]")
        return
    for i, c in enumerate(chunks, 1):
        title_parts = [f"[{i}]", c.source_type]
        if c.source_file:
            title_parts.append(c.source_file)
        if c.confidence:
            title_parts.append(f"({c.confidence})")
        if c.distance == float("inf"):
            reason = c.metadata.get("_retrieval_reason") or "linked"
            title_parts.append(reason)
            border = "yellow"
        else:
            title_parts.append(f"dist={c.distance:.3f}")
            border = "blue"
        console.print(
            Panel(c.text, title=" ".join(title_parts), border_style=border)
        )


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted.[/dim]")
        sys.exit(130)
